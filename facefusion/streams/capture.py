from threading import Lock, Thread
from time import sleep
from typing import Optional, Tuple

import cv2

from facefusion.types import VisionFrame


class LatestFrameCapture:
	def __init__(self, camera_capture : cv2.VideoCapture) -> None:
		self.camera_capture = camera_capture
		self.lock = Lock()
		self.latest_frame : Optional[VisionFrame] = None
		self.latest_frame_number = 0
		self.read_frame_number = 0
		self.is_running = camera_capture.isOpened()
		self.reader_thread = Thread(target = self.read_frames, daemon = True)
		self.reader_thread.start()

	def read_frames(self) -> None:
		while self.is_running and self.camera_capture.isOpened():
			has_frame, frame = self.camera_capture.read()

			if has_frame:
				with self.lock:
					self.latest_frame = frame
					self.latest_frame_number += 1
			else:
				sleep(0.01)

	def read(self) -> Tuple[bool, Optional[VisionFrame]]:
		with self.lock:
			if self.latest_frame is not None and self.latest_frame_number != self.read_frame_number:
				self.read_frame_number = self.latest_frame_number
				return True, self.latest_frame.copy()

		sleep(0.01)
		return False, None

	def set(self, prop_id : int, value : float) -> bool:
		return self.camera_capture.set(prop_id, value)

	def isOpened(self) -> bool:
		return self.is_running and self.camera_capture.isOpened()

	def release(self) -> None:
		self.is_running = False

		if self.reader_thread.is_alive():
			self.reader_thread.join(timeout = 1)

		self.camera_capture.release()
