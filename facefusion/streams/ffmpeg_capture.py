import subprocess
from threading import Lock, Thread
from time import monotonic, sleep
from typing import Dict, List, Optional, Tuple

import numpy

from facefusion.types import VisionFrame


class FFmpegCapture:
	def __init__(self, url : str, width : int, height : int) -> None:
		self.width = width
		self.height = height
		self.frame_size = width * height * 3
		self.ffmpeg_command = self.create_ffmpeg_command(url, width, height)
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

	def open_ffmpeg(self, url : str, width : int, height : int) -> subprocess.Popen[bytes]:
		return subprocess.Popen(self.ffmpeg_command, stdout = subprocess.PIPE, stderr = subprocess.DEVNULL)

	def create_ffmpeg_command(self, url : str, width : int, height : int) -> List[str]:
		return\
		[
			'ffmpeg',
			'-hide_banner',
			'-loglevel', 'error',
			'-fflags', 'nobuffer',
			'-flags', 'low_delay',
			'-i', url,
			'-an',
			'-vf', 'scale=' + str(width) + ':' + str(height),
			'-pix_fmt', 'bgr24',
			'-f', 'rawvideo',
			'pipe:1'
		]

	def read_frames(self) -> None:
		while self.is_running and self.process.poll() is None and self.process.stdout is not None:
			frame_buffer = self.read_frame_buffer()

			if len(frame_buffer) != self.frame_size:
				break

			frame = numpy.frombuffer(frame_buffer, dtype = numpy.uint8).reshape((self.height, self.width, 3))

			with self.lock:
				self.latest_frame = frame
				self.latest_frame_number += 1

		if self.is_running and self.process.poll() is None:
			self.process.terminate()

		self.is_running = False

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
			'max_gap_ms': max_gap_ms
		}
