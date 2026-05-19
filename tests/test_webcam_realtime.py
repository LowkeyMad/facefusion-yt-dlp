from typing import Any, Dict, List, Tuple
from unittest.mock import ANY, patch

import numpy

from facefusion import streamer
from facefusion.streamer import restrict_realtime_resolution
from facefusion.types import VisionFrame
from facefusion.uis.components import webcam, webcam_options


class FakeCameraCapture:
	def __init__(self) -> None:
		self.properties : List[Tuple[int, float]] = []

	def isOpened(self) -> bool:
		return True

	def set(self, prop_id : int, value : float) -> bool:
		self.properties.append((prop_id, value))
		return True


class OneFrameCapture:
	def __init__(self, frame : VisionFrame) -> None:
		self.frame = frame
		self.has_frame = True
		self.is_opened = True

	def isOpened(self) -> bool:
		return self.is_opened

	def read(self) -> Tuple[bool, VisionFrame]:
		if self.has_frame:
			self.has_frame = False
			return True, self.frame

		self.is_opened = False
		return False, self.frame

	def release(self) -> None:
		self.is_opened = False


class FakeProgress:
	def __enter__(self) -> 'FakeProgress':
		return self

	def __exit__(self, *args : object) -> None:
		pass

	def update(self) -> None:
		pass


def create_vision_frame(width : int, height : int) -> VisionFrame:
	return numpy.ones((height, width, 3), dtype = numpy.uint8)


def test_restrict_realtime_resolution() -> None:
	assert restrict_realtime_resolution(1920, 1080) == (640, 360)
	assert restrict_realtime_resolution(1280, 720) == (640, 360)
	assert restrict_realtime_resolution(640, 360) == (640, 360)
	assert restrict_realtime_resolution(426, 240) == (426, 240)


def test_webcam_realtime_mode_option_defaults_enabled() -> None:
	registered_components : Dict[str, Any] = {}

	def register_component(name : str, component : Any) -> Any:
		return registered_components.setdefault(name, component)

	with patch('facefusion.uis.components.webcam_options.detect_local_camera_ids', return_value = [ 0 ]), patch('facefusion.uis.components.webcam_options.register_ui_component', side_effect = register_component):
		webcam_options.render()

	realtime_mode_checkbox = registered_components.get('webcam_realtime_mode_checkbox')

	assert realtime_mode_checkbox is not None
	assert realtime_mode_checkbox.label == 'REALTIME MODE'
	assert realtime_mode_checkbox.value is True


def test_start_remote_realtime_mode_uses_restricted_capture_size() -> None:
	camera_capture = FakeCameraCapture()

	with patch('facefusion.uis.components.webcam.state_manager.init_item'), patch('facefusion.uis.components.webcam.state_manager.sync_state'), patch('facefusion.uis.components.webcam.state_manager.get_item', return_value = [ 'face_swapper' ]), patch('facefusion.uis.components.webcam.prepare_youtube_cookies_path', return_value = None), patch('facefusion.uis.components.webcam.resolve_stream_url', return_value = 'https://stream.example.com/live.m3u8'), patch('facefusion.uis.components.webcam.get_remote_camera_capture', return_value = camera_capture) as get_remote_camera_capture, patch('facefusion.uis.components.webcam.process_latest_capture', return_value = iter([ create_vision_frame(640, 360) ])) as process_latest_capture:
		capture_vision_frame = next(webcam.start(0, 'https://www.youtube.com/watch?v=test', None, False, True, 'inline', '1920x1080', 30))

	get_remote_camera_capture.assert_called_once_with('https://stream.example.com/live.m3u8', 640, 360)
	process_latest_capture.assert_called_once_with(camera_capture, 30, True, ANY)
	assert capture_vision_frame.shape == (1080, 1920, 3)


def test_start_remote_realtime_mode_disabled_uses_selected_capture_size() -> None:
	camera_capture = FakeCameraCapture()

	with patch('facefusion.uis.components.webcam.state_manager.init_item'), patch('facefusion.uis.components.webcam.state_manager.sync_state'), patch('facefusion.uis.components.webcam.state_manager.get_item', return_value = [ 'face_swapper' ]), patch('facefusion.uis.components.webcam.prepare_youtube_cookies_path', return_value = None), patch('facefusion.uis.components.webcam.resolve_stream_url', return_value = 'https://stream.example.com/live.m3u8'), patch('facefusion.uis.components.webcam.get_remote_camera_capture', return_value = camera_capture) as get_remote_camera_capture, patch('facefusion.uis.components.webcam.process_latest_capture', return_value = iter([ create_vision_frame(1920, 1080) ])) as process_latest_capture:
		capture_vision_frame = next(webcam.start(0, 'https://www.youtube.com/watch?v=test', None, False, False, 'inline', '1920x1080', 30))

	get_remote_camera_capture.assert_called_once_with('https://stream.example.com/live.m3u8', 1920, 1080)
	process_latest_capture.assert_called_once_with(camera_capture, 30, False, ANY)
	assert capture_vision_frame.shape == (1080, 1920, 3)


def test_start_remote_without_processors_uses_raw_capture() -> None:
	camera_capture = FakeCameraCapture()

	with patch('facefusion.uis.components.webcam.state_manager.init_item'), patch('facefusion.uis.components.webcam.state_manager.sync_state'), patch('facefusion.uis.components.webcam.state_manager.get_item', return_value = []), patch('facefusion.uis.components.webcam.prepare_youtube_cookies_path', return_value = None), patch('facefusion.uis.components.webcam.resolve_stream_url', return_value = 'https://stream.example.com/live.m3u8'), patch('facefusion.uis.components.webcam.get_remote_camera_capture', return_value = camera_capture) as get_remote_camera_capture, patch('facefusion.uis.components.webcam.process_raw_latest_capture', return_value = iter([ create_vision_frame(640, 360) ])) as process_raw_latest_capture, patch('facefusion.uis.components.webcam.process_latest_capture') as process_latest_capture:
		capture_vision_frame = next(webcam.start(0, 'https://www.youtube.com/watch?v=test', None, False, True, 'inline', '1920x1080', 30))

	get_remote_camera_capture.assert_called_once_with('https://stream.example.com/live.m3u8', 640, 360)
	process_raw_latest_capture.assert_called_once_with(camera_capture, ANY)
	process_latest_capture.assert_not_called()
	assert capture_vision_frame.shape == (360, 640, 3)


def test_start_remote_preview_stream_only_uses_selected_capture_size() -> None:
	camera_capture = FakeCameraCapture()

	with patch('facefusion.uis.components.webcam.state_manager.init_item'), patch('facefusion.uis.components.webcam.state_manager.sync_state'), patch('facefusion.uis.components.webcam.prepare_youtube_cookies_path', return_value = None), patch('facefusion.uis.components.webcam.resolve_stream_url', return_value = 'https://stream.example.com/live.m3u8'), patch('facefusion.uis.components.webcam.get_remote_camera_capture', return_value = camera_capture) as get_remote_camera_capture, patch('facefusion.uis.components.webcam.process_raw_latest_capture', return_value = iter([ create_vision_frame(1920, 1080) ])) as process_raw_latest_capture, patch('facefusion.uis.components.webcam.process_latest_capture') as process_latest_capture:
		capture_vision_frame = next(webcam.start(0, 'https://www.youtube.com/watch?v=test', None, True, True, 'inline', '1920x1080', 30))

	get_remote_camera_capture.assert_called_once_with('https://stream.example.com/live.m3u8', 1920, 1080)
	process_raw_latest_capture.assert_called_once_with(camera_capture, ANY)
	process_latest_capture.assert_not_called()
	assert capture_vision_frame.shape == (1080, 1920, 3)


def test_process_latest_capture_realtime_mode_processes_restricted_frame() -> None:
	camera_capture = OneFrameCapture(create_vision_frame(1920, 1080))

	def assert_restricted_frame(vision_frame : VisionFrame, *args : object) -> VisionFrame:
		assert vision_frame.shape == (360, 640, 3)
		return vision_frame

	with patch('facefusion.streamer.read_static_images', return_value = []), patch('facefusion.streamer.create_empty_audio_frame', return_value = None), patch('facefusion.streamer.get_processors_modules', return_value = []), patch('facefusion.streamer.state_manager.get_item', return_value = 'error'), patch('facefusion.streamer.tqdm', return_value = FakeProgress()), patch('facefusion.streamer.analyse_stream', return_value = False) as analyse_stream, patch('facefusion.streamer.process_stream_frame', side_effect = assert_restricted_frame) as process_stream_frame:
		capture_vision_frame = next(streamer.process_latest_capture(camera_capture, 30, True))

	analyse_stream.assert_called_once()
	process_stream_frame.assert_called_once()
	assert capture_vision_frame.shape == (360, 640, 3)


def test_start_local_webcam_uses_existing_local_capture_path() -> None:
	camera_capture = FakeCameraCapture()

	with patch('facefusion.uis.components.webcam.state_manager.init_item'), patch('facefusion.uis.components.webcam.state_manager.sync_state'), patch('facefusion.uis.components.webcam.state_manager.get_item', return_value = [ 'face_swapper' ]), patch('facefusion.uis.components.webcam.get_local_camera_capture', return_value = camera_capture) as get_local_camera_capture, patch('facefusion.uis.components.webcam.get_remote_camera_capture') as get_remote_camera_capture, patch('facefusion.uis.components.webcam.multi_process_capture', return_value = iter([ create_vision_frame(1920, 1080) ])) as multi_process_capture, patch('facefusion.uis.components.webcam.process_latest_capture') as process_latest_capture:
		capture_vision_frame = next(webcam.start(0, '', None, False, True, 'inline', '1920x1080', 30))

	get_local_camera_capture.assert_called_once_with(0)
	get_remote_camera_capture.assert_not_called()
	multi_process_capture.assert_called_once_with(camera_capture, 30, ANY)
	process_latest_capture.assert_not_called()
	assert capture_vision_frame.shape == (1080, 1920, 3)
