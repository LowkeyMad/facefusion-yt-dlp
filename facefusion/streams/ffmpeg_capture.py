import subprocess
from typing import Optional, Tuple

import numpy

from facefusion.types import VisionFrame


class FFmpegCapture:
	def __init__(self, url : str, width : int, height : int) -> None:
		self.width = width
		self.height = height
		self.frame_size = width * height * 3
		self.process = self.open_ffmpeg(url, width, height)

	def open_ffmpeg(self, url : str, width : int, height : int) -> subprocess.Popen[bytes]:
		commands =\
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
		return subprocess.Popen(commands, stdout = subprocess.PIPE, stderr = subprocess.DEVNULL)

	def read(self) -> Tuple[bool, Optional[VisionFrame]]:
		if not self.isOpened() or self.process.stdout is None:
			return False, None

		frame_buffer = self.read_frame_buffer()

		if len(frame_buffer) != self.frame_size:
			self.release()
			return False, None

		frame = numpy.frombuffer(frame_buffer, dtype = numpy.uint8).reshape((self.height, self.width, 3))
		return True, frame.copy()

	def read_frame_buffer(self) -> bytes:
		frame_buffer = bytearray()

		while len(frame_buffer) < self.frame_size and self.process.stdout is not None:
			chunk = self.process.stdout.read(self.frame_size - len(frame_buffer))

			if not chunk:
				break

			frame_buffer.extend(chunk)

		return bytes(frame_buffer)

	def set(self, prop_id : int, value : float) -> bool:
		return True

	def isOpened(self) -> bool:
		return self.process.poll() is None

	def release(self) -> None:
		if self.process.poll() is None:
			self.process.terminate()

			try:
				self.process.wait(timeout = 1)
			except subprocess.TimeoutExpired:
				self.process.kill()
				self.process.wait()
