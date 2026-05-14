import subprocess
from time import sleep, time
from typing import List, Optional
from unittest.mock import patch

import numpy

from facefusion.streams.ffmpeg_capture import FFmpegCapture


class FakeStdout:
	def __init__(self, chunks : List[bytes]) -> None:
		self.chunks = chunks

	def read(self, size : int) -> bytes:
		if self.chunks:
			return self.chunks.pop(0)

		return bytes()


class FakeProcess:
	def __init__(self, chunks : List[bytes]) -> None:
		self.stdout : Optional[FakeStdout] = FakeStdout(chunks)
		self.is_running = True
		self.is_terminated = False
		self.is_killed = False

	def poll(self) -> Optional[int]:
		return None if self.is_running else 0

	def terminate(self) -> None:
		self.is_running = False
		self.is_terminated = True

	def kill(self) -> None:
		self.is_running = False
		self.is_killed = True

	def wait(self, timeout : Optional[int] = None) -> int:
		self.is_running = False
		return 0


def create_frame_buffer(width : int, height : int, value : int) -> bytes:
	return numpy.full((height, width, 3), value, dtype = numpy.uint8).tobytes()


def wait_for_reader(capture : FFmpegCapture, latest_frame_number : int) -> None:
	timeout = time() + 1

	while capture.latest_frame_number < latest_frame_number and time() < timeout:
		sleep(0.01)


def test_ffmpeg_capture_reads_newest_frame_from_background_reader() -> None:
	frame_buffers =\
	[
		create_frame_buffer(2, 1, 1),
		create_frame_buffer(2, 1, 2)
	]
	fake_process = FakeProcess(frame_buffers)

	with patch('facefusion.streams.ffmpeg_capture.subprocess.Popen', return_value = fake_process) as popen:
		capture = FFmpegCapture('https://stream.example.com/live.m3u8', 2, 1)
		wait_for_reader(capture, 2)
		has_frame, frame = capture.read()
		has_new_frame, new_frame = capture.read()
		capture.release()

	popen.assert_called_once_with([ 'ffmpeg', '-hide_banner', '-loglevel', 'error', '-fflags', 'nobuffer', '-flags', 'low_delay', '-re', '-i', 'https://stream.example.com/live.m3u8', '-an', '-vf', 'scale=2:1', '-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1' ], stdout = subprocess.PIPE, stderr = subprocess.DEVNULL)
	assert has_frame is True
	assert frame is not None
	assert frame[0, 0, 0] == 2
	assert has_new_frame is False
	assert new_frame is None
	assert fake_process.is_terminated is True
