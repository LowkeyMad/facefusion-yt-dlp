import os
import shutil
from typing import Iterator, List, Optional, Tuple

import cv2
import gradio

from facefusion import logger, state_manager, translator
from facefusion.camera_manager import clear_camera_pool, get_local_camera_capture, get_remote_camera_capture
from facefusion.filesystem import create_directory, has_image, is_file
from facefusion.streamer import multi_process_capture, open_stream, process_latest_capture, process_raw_latest_capture
from facefusion.streams.ytdlp import resolve_stream_url
from facefusion.types import Fps, VisionFrame, WebcamMode
from facefusion.uis.core import get_ui_component
from facefusion.uis.types import File
from facefusion.vision import fit_cover_frame, unpack_resolution

SOURCE_FILE : Optional[gradio.File] = None
WEBCAM_IMAGE : Optional[gradio.Image] = None
WEBCAM_START_BUTTON : Optional[gradio.Button] = None
WEBCAM_STOP_BUTTON : Optional[gradio.Button] = None


def render() -> None:
	global SOURCE_FILE
	global WEBCAM_IMAGE
	global WEBCAM_START_BUTTON
	global WEBCAM_STOP_BUTTON

	has_source_image = has_image(state_manager.get_item('source_paths'))
	SOURCE_FILE = gradio.File(
		label = translator.get('uis.source_file'),
		file_count = 'multiple',
		value = state_manager.get_item('source_paths') if has_source_image else None
	)
	WEBCAM_IMAGE = gradio.Image(
		label = translator.get('uis.webcam_image'),
		format = 'jpeg',
		visible = False
	)
	WEBCAM_START_BUTTON = gradio.Button(
		value = translator.get('uis.start_button'),
		variant = 'primary',
		size = 'sm'
	)
	WEBCAM_STOP_BUTTON = gradio.Button(
		value = translator.get('uis.stop_button'),
		size = 'sm',
		visible = False
	)


def listen() -> None:
	SOURCE_FILE.change(update_source, inputs = SOURCE_FILE, outputs = SOURCE_FILE)
	webcam_device_id_dropdown = get_ui_component('webcam_device_id_dropdown')
	webcam_stream_url_textbox = get_ui_component('webcam_stream_url_textbox')
	webcam_youtube_cookies_file = get_ui_component('webcam_youtube_cookies_file')
	webcam_preview_stream_only_checkbox = get_ui_component('webcam_preview_stream_only_checkbox')
	webcam_mode_radio = get_ui_component('webcam_mode_radio')
	webcam_resolution_dropdown = get_ui_component('webcam_resolution_dropdown')
	webcam_fps_slider = get_ui_component('webcam_fps_slider')

	if webcam_device_id_dropdown and webcam_stream_url_textbox and webcam_youtube_cookies_file and webcam_preview_stream_only_checkbox and webcam_mode_radio and webcam_resolution_dropdown and webcam_fps_slider:
		WEBCAM_START_BUTTON.click(pre_start, outputs = [ SOURCE_FILE, WEBCAM_IMAGE, WEBCAM_START_BUTTON, WEBCAM_STOP_BUTTON ])
		start_event = WEBCAM_START_BUTTON.click(start, inputs = [ webcam_device_id_dropdown, webcam_stream_url_textbox, webcam_youtube_cookies_file, webcam_preview_stream_only_checkbox, webcam_mode_radio, webcam_resolution_dropdown, webcam_fps_slider ], outputs = WEBCAM_IMAGE)
		start_event.then(pre_stop)
		WEBCAM_STOP_BUTTON.click(stop, cancels = start_event, outputs = WEBCAM_IMAGE)
		WEBCAM_STOP_BUTTON.click(pre_stop, outputs = [ SOURCE_FILE, WEBCAM_IMAGE, WEBCAM_START_BUTTON, WEBCAM_STOP_BUTTON ])


def update_source(files : List[File]) -> gradio.File:
	file_names = [ file.name for file in files ] if files else None
	has_source_image = has_image(file_names)

	if has_source_image:
		state_manager.set_item('source_paths', file_names)
		return gradio.File(value = file_names)

	state_manager.clear_item('source_paths')
	return gradio.File(value = None)


def pre_start() -> Tuple[gradio.File, gradio.Image, gradio.Button, gradio.Button]:
	return gradio.File(visible = False), gradio.Image(visible = True), gradio.Button(visible = False), gradio.Button(visible = True)


def pre_stop() -> Tuple[gradio.File, gradio.Image, gradio.Button, gradio.Button]:
	return gradio.File(visible = True), gradio.Image(visible = False), gradio.Button(visible = True), gradio.Button(visible = False)


def start(webcam_device_id : int, webcam_stream_url : str, youtube_cookies_file : File, preview_stream_only : bool, webcam_mode : WebcamMode, webcam_resolution : str, webcam_fps : Fps) -> Iterator[VisionFrame]:
	state_manager.init_item('face_selector_mode', 'one')
	state_manager.sync_state()

	camera_capture = None
	webcam_stream_url = webcam_stream_url.strip() if webcam_stream_url else None

	if webcam_stream_url:
		youtube_cookies_path = prepare_youtube_cookies_path(youtube_cookies_file)
		stream_url = resolve_stream_url(webcam_stream_url, youtube_cookies_path)
		if stream_url:
			camera_capture = get_remote_camera_capture(stream_url)
		else:
			logger.error(translator.get('webcam_stream_not_resolved'), __name__)
	else:
		camera_capture = get_local_camera_capture(webcam_device_id)

	webcam_width, webcam_height = unpack_resolution(webcam_resolution)

	if camera_capture and camera_capture.isOpened():
		stream = None

		if webcam_mode in [ 'udp', 'v4l2' ]:
			stream = open_stream(webcam_mode, webcam_resolution, webcam_fps) #type:ignore[arg-type]

		camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
		camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)
		camera_capture.set(cv2.CAP_PROP_FPS, webcam_fps)

		if preview_stream_only:
			capture_vision_frames = process_raw_latest_capture(camera_capture)
		else:
			capture_vision_frames = process_latest_capture(camera_capture, webcam_fps) if webcam_stream_url else multi_process_capture(camera_capture, webcam_fps)

		for capture_vision_frame in capture_vision_frames:
			capture_vision_frame = cv2.cvtColor(capture_vision_frame, cv2.COLOR_BGR2RGB)
			capture_vision_frame = fit_cover_frame(capture_vision_frame, (webcam_width, webcam_height))

			if webcam_mode == 'inline':
				yield capture_vision_frame
			if webcam_mode in [ 'udp', 'v4l2' ]:
				try:
					stream.stdin.write(capture_vision_frame.tobytes())
				except Exception:
					pass


def stop() -> gradio.Image:
	clear_camera_pool()
	return gradio.Image(value = None)


def prepare_youtube_cookies_path(youtube_cookies_file : File) -> Optional[str]:
	if youtube_cookies_file and is_file(youtube_cookies_file.name):
		cookies_directory_path = os.path.join(state_manager.get_item('temp_path'), 'facefusion', 'youtube')
		cookies_path = os.path.join(cookies_directory_path, 'cookies.txt')

		if create_directory(cookies_directory_path):
			shutil.copy(youtube_cookies_file.name, cookies_path)

			if is_file(cookies_path):
				return cookies_path

	return None
