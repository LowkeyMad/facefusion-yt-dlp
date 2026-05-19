import subprocess
from collections import deque
from functools import lru_cache
from threading import Lock, Thread
from time import monotonic, sleep
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy

from facefusion.types import VisionFrame


class FFmpegCapture:
	def __init__(self, url : str, width : int, height : int, restore_default_buffering : bool = False, stream_fps : float = 30.0, stream_delay : float = 0.0) -> None:
		self.width = width
		self.height = height
		self.frame_size = width * height * 3
		self.restore_default_buffering = restore_default_buffering
		self.decode_mode = 'cpu'
		self.stream_fps = max(stream_fps, 1.0)
		self.stream_delay = max(stream_delay, 0.0)
		self.buffer_target_frame_total = max(1, int(round(self.stream_fps * self.stream_delay))) if self.stream_delay > 0 else 0
		self.buffer_frame_total = max(1, self.buffer_target_frame_total)
		self.frame_buffer_queue : Deque[VisionFrame] = deque()
		self.dropped_frame_total = 0
		self.underrun_total = 0
		self.is_buffer_ready = self.stream_delay == 0
		self.ffmpeg_command = self.create_ffmpeg_command(url, width, height, restore_default_buffering)
		self.process = self.open_ffmpeg(url, width, height)
		self.lock = Lock()
		self.latest_frame : Optional[VisionFrame] = None
		self.latest_frame_number = 0
		self.read_frame_number = 0
		self.read_count = 0
		self.read_miss_count = 0
		self.read_gap_times : List[float] = []
		self.last_successful_read_at : Optional[float] = None
		self.is_running = self.process.poll() is None
		self.reader_thread = Thread(target = self.read_frames, daemon = True)
		self.reader_thread.start()
		self.publisher_thread : Optional[Thread] = None

		if self.stream_delay > 0:
			self.publisher_thread = Thread(target = self.publish_frames, daemon = True)
			self.publisher_thread.start()

	def open_ffmpeg(self, url : str, width : int, height : int) -> subprocess.Popen[bytes]:
		return subprocess.Popen(self.ffmpeg_command, stdout = subprocess.PIPE, stderr = subprocess.DEVNULL)

	def create_ffmpeg_command(self, url : str, width : int, height : int, restore_default_buffering : bool = False) -> List[str]:
		commands =\
		[
			'ffmpeg',
			'-hide_banner',
			'-loglevel', 'error'
		]

		if not restore_default_buffering:
			commands.extend(
			[
				'-fflags', 'nobuffer',
				'-flags', 'low_delay'
			])

		commands.extend(
		[
			'-i', url,
			'-an',
			'-vf', 'scale=' + str(width) + ':' + str(height),
			'-pix_fmt', 'bgr24',
			'-f', 'rawvideo',
			'pipe:1'
		])
		return commands

	def read_frames(self) -> None:
		while self.is_running and self.process.poll() is None and self.process.stdout is not None:
			frame_buffer = self.read_frame_buffer()

			if len(frame_buffer) != self.frame_size:
				break

			frame = numpy.frombuffer(frame_buffer, dtype = numpy.uint8).reshape((self.height, self.width, 3))

			with self.lock:
				if self.stream_delay > 0:
					self.enqueue_frame(frame)
				else:
					self.latest_frame = frame
					self.latest_frame_number += 1

		if self.is_running and self.process.poll() is None:
			self.process.terminate()

		self.is_running = False

	def publish_frames(self) -> None:
		publish_interval = 1.0 / self.stream_fps

		while self.is_running and self.process.poll() is None:
			publish_started_at = monotonic()
			frame = None

			with self.lock:
				if not self.is_buffer_ready and len(self.frame_buffer_queue) >= self.buffer_target_frame_total:
					self.is_buffer_ready = True

				if self.is_buffer_ready:
					frame = self.publish_buffered_frame()

			if frame is not None:
				with self.lock:
					self.latest_frame = frame
					self.latest_frame_number += 1

			sleep(max(0.0, publish_interval - (monotonic() - publish_started_at)))

	def enqueue_frame(self, frame : VisionFrame) -> None:
		self.frame_buffer_queue.append(frame.copy())

		while len(self.frame_buffer_queue) > self.buffer_frame_total:
			self.frame_buffer_queue.popleft()
			self.dropped_frame_total += 1

	def publish_buffered_frame(self) -> Optional[VisionFrame]:
		if self.frame_buffer_queue:
			return self.frame_buffer_queue.popleft()

		self.underrun_total += 1
		self.is_buffer_ready = False
		return None

	def read(self) -> Tuple[bool, Optional[VisionFrame]]:
		with self.lock:
			if self.latest_frame is not None and self.latest_frame_number != self.read_frame_number:
				now = monotonic()
				self.read_count += 1

				if self.last_successful_read_at is not None:
					self.read_gap_times.append(now - self.last_successful_read_at)

				self.last_successful_read_at = now
				self.read_frame_number = self.latest_frame_number
				return True, self.latest_frame.copy()

			self.read_miss_count += 1

		sleep(0.01)
		return False, None

	def read_frame_buffer(self) -> bytes:
		frame_buffer = bytearray()

		while self.is_running and len(frame_buffer) < self.frame_size and self.process.stdout is not None:
			chunk = self.process.stdout.read(self.frame_size - len(frame_buffer))

			if not chunk:
				break

			frame_buffer.extend(chunk)

		return bytes(frame_buffer)

	def set(self, prop_id : int, value : float) -> bool:
		return True

	def isOpened(self) -> bool:
		return self.is_running and self.process.poll() is None

	def release(self) -> None:
		self.is_running = False

		if self.process.poll() is None:
			self.process.terminate()

			try:
				self.process.wait(timeout = 1)
			except subprocess.TimeoutExpired:
				self.process.kill()
				self.process.wait()

		if self.reader_thread.is_alive():
			self.reader_thread.join(timeout = 1)

		if self.publisher_thread and self.publisher_thread.is_alive():
			self.publisher_thread.join(timeout = 1)

	def get_read_gap_stats(self) -> Dict[str, Optional[float]]:
		if self.read_gap_times:
			avg_gap_ms = 1000 * sum(self.read_gap_times) / len(self.read_gap_times)
			max_gap_ms = 1000 * max(self.read_gap_times)
		else:
			avg_gap_ms = None
			max_gap_ms = None

		return\
		{
			'reads': self.read_count,
			'misses': self.read_miss_count,
			'avg_gap_ms': avg_gap_ms,
			'max_gap_ms': max_gap_ms,
			'stream_delay': self.stream_delay,
			'buffer_target_frames': self.buffer_target_frame_total,
			'buffer_length': len(self.frame_buffer_queue),
			'dropped_frames': self.dropped_frame_total,
			'underruns': self.underrun_total
		}

	def get_ingest_diagnostics(self) -> Dict[str, Any]:
		return\
		{
			'buffering': 'default_probe_analyze' if self.restore_default_buffering else 'low_delay_nobuffer',
			'decode_mode': self.decode_mode,
			'stream_delay': self.stream_delay,
			'buffer_target_frames': self.buffer_target_frame_total,
			'buffer_length': len(self.frame_buffer_queue),
			'dropped_frames': self.dropped_frame_total,
			'underruns': self.underrun_total,
			'h264_cuvid_available': has_ffmpeg_decoder('h264_cuvid'),
			'command_variants':
			{
				'low_delay_nobuffer': self.create_ffmpeg_command('<stream_url>', self.width, self.height, False),
				'default_probe_analyze': self.create_ffmpeg_command('<stream_url>', self.width, self.height, True)
			}
		}


@lru_cache(maxsize = 1)
def has_ffmpeg_decoder(decoder_name : str) -> bool:
	try:
		process = subprocess.run([ 'ffmpeg', '-hide_banner', '-decoders' ], capture_output = True, text = True, timeout = 5)
	except Exception:
		return False

	return process.returncode == 0 and decoder_name in process.stdout
