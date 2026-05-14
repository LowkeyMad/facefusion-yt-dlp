import subprocess
from threading import Lock, Thread
from time import sleep
from typing import Optional, Tuple

import numpy

from facefusion.types import VisionFrame


class FFmpegCapture:
	def __init__(self, url : str, width : int, height : int) -> None:
		self.width = width
		self.height = height
		self.frame_size = width * height * 3
		self.process = self.open_ffmpeg(url, width, height)
		self.lock = Lock()
		self.latest_frame : Optional[VisionFrame] = None
		self.latest_frame_number = 0
		self.read_frame_number = 0
		self.is_running = self.process.poll() is None
		self.reader_thread = Thread(target = self.read_frames, daemon = True)
		self.reader_thread.start()

	def open_ffmpeg(self, url : str, width : int, height : int) -> subprocess.Popen[bytes]:
		commands =\
		[
			'ffmpeg',
			'-hide_banner',
			'-loglevel', 'error',
			'-fflags', 'nobuffer',
			'-flags', 'low_delay',
			'-re',
			'-i', url,
			'-an',
			'-vf', 'scale=' + str(width) + ':' + str(height),
			'-pix_fmt', 'bgr24',
			'-f', 'rawvideo',
			'pipe:1'
		]
		return subprocess.Popen(commands, stdout = subprocess.PIPE, stderr = subprocess.DEVNULL)

	def read_frames(self) -> None:
		while self.is_running and self.process.poll() is None and self.process.stdout is not None:
			frame_buffer = self.read_frame_buffer()

			if len(frame_buffer) != self.frame_size:
				break

			frame = numpy.frombuffer(frame_buffer, dtype = numpy.uint8).reshape((self.height, self.width, 3))

			with self.lock:
				self.latest_frame = frame.copy()
				self.latest_frame_number += 1

		if self.is_running and self.process.poll() is None:
			self.process.terminate()

		self.is_running = False

	def read(self) -> Tuple[bool, Optional[VisionFrame]]:
		with self.lock:
			if self.latest_frame is not None and self.latest_frame_number != self.read_frame_number:
				self.read_frame_number = self.latest_frame_number
				return True, self.latest_frame.copy()

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
