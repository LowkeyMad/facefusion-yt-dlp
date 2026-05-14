import os
import subprocess
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from time import sleep
from types import ModuleType
from typing import Deque, Dict, Iterator, List, Optional

import cv2
import numpy
from tqdm import tqdm

from facefusion import ffmpeg_builder, logger, state_manager, translator
from facefusion.audio import create_empty_audio_frame
from facefusion.content_analyser import analyse_stream
from facefusion.face_analyser import get_many_faces
from facefusion.ffmpeg import open_ffmpeg
from facefusion.filesystem import is_directory
from facefusion.processors.core import get_processors_modules
from facefusion.types import AudioFrame, Face, Fps, StreamMode, VisionFrame
from facefusion.vision import extract_vision_mask, read_static_images

REALTIME_PROCESS_WIDTH = 640
FACE_DEPENDENT_PROCESSORS =\
{
	'age_modifier',
	'deep_swapper',
	'expression_restorer',
	'face_debugger',
	'face_editor',
	'face_enhancer',
	'face_swapper',
	'lip_syncer'
}


def multi_process_capture(camera_capture : cv2.VideoCapture, camera_fps : Fps) -> Iterator[VisionFrame]:
	capture_deque : Deque[VisionFrame] = deque()

	with tqdm(desc = translator.get('streaming'), unit = 'frame', disable = state_manager.get_item('log_level') in [ 'warn', 'error' ]) as progress:
		with ThreadPoolExecutor(max_workers = state_manager.get_item('execution_thread_count')) as executor:
			futures = []

			while camera_capture and camera_capture.isOpened():
				has_capture_vision_frame, capture_vision_frame = camera_capture.read()

				if not has_capture_vision_frame:
					break

				if analyse_stream(capture_vision_frame, camera_fps):
					camera_capture.release()

				if numpy.any(capture_vision_frame):
					future = executor.submit(process_stream_frame, capture_vision_frame)
					futures.append(future)

				for future_done in [ future for future in futures if future.done() ]:
					capture_vision_frame = future_done.result()
					capture_deque.append(capture_vision_frame)
					futures.remove(future_done)

				while capture_deque:
					progress.update()
					yield capture_deque.popleft()


def process_latest_capture(camera_capture : cv2.VideoCapture, camera_fps : Fps) -> Iterator[VisionFrame]:
	source_vision_frames = read_static_images(state_manager.get_item('source_paths'))
	source_audio_frame = create_empty_audio_frame()
	source_voice_frame = create_empty_audio_frame()
	processor_modules = []
	source_faces : Dict[str, Face] = {}

	for processor_module in get_processors_modules(state_manager.get_item('processors')):
		logger.disable()
		if processor_module.pre_process('stream'):
			processor_modules.append(processor_module)
			if hasattr(processor_module, 'extract_source_face'):
				source_face = processor_module.extract_source_face(source_vision_frames)
				if source_face:
					source_faces[processor_module.__name__] = source_face
		logger.enable()

	skip_no_face_frames = bool(processor_modules) and all(processor_module.__name__.rsplit('.', 1)[-1] in FACE_DEPENDENT_PROCESSORS for processor_module in processor_modules)

	with tqdm(desc = translator.get('streaming'), unit = 'frame', disable = state_manager.get_item('log_level') in [ 'warn', 'error' ]) as progress:
		while camera_capture and camera_capture.isOpened():
			has_capture_vision_frame, capture_vision_frame = camera_capture.read()

			if not has_capture_vision_frame:
				sleep(0.01)
				continue

			if analyse_stream(capture_vision_frame, camera_fps):
				camera_capture.release()
				break

			if numpy.any(capture_vision_frame):
				capture_vision_frame = resize_realtime_frame(capture_vision_frame)

				if skip_no_face_frames and not get_many_faces([ capture_vision_frame ]):
					progress.update()
					yield capture_vision_frame
					continue

				progress.update()
				yield process_stream_frame(capture_vision_frame, source_vision_frames, source_audio_frame, source_voice_frame, processor_modules, source_faces)


def resize_realtime_frame(vision_frame : VisionFrame) -> VisionFrame:
	height, width = vision_frame.shape[:2]

	if width > REALTIME_PROCESS_WIDTH:
		realtime_height = int(height * REALTIME_PROCESS_WIDTH / width)
		return cv2.resize(vision_frame, (REALTIME_PROCESS_WIDTH, realtime_height), interpolation = cv2.INTER_AREA)

	return vision_frame


def process_stream_frame(target_vision_frame : VisionFrame, source_vision_frames : Optional[List[VisionFrame]] = None, source_audio_frame : Optional[AudioFrame] = None, source_voice_frame : Optional[AudioFrame] = None, processor_modules : Optional[List[ModuleType]] = None, source_faces : Optional[Dict[str, Face]] = None) -> VisionFrame:
	source_vision_frames = source_vision_frames or read_static_images(state_manager.get_item('source_paths'))
	source_audio_frame = source_audio_frame if source_audio_frame is not None else create_empty_audio_frame()
	source_voice_frame = source_voice_frame if source_voice_frame is not None else create_empty_audio_frame()
	check_processor_modules = processor_modules is None
	processor_modules = processor_modules or get_processors_modules(state_manager.get_item('processors'))
	temp_vision_frame = target_vision_frame.copy()
	temp_vision_mask = extract_vision_mask(temp_vision_frame)

	for processor_module in processor_modules:
		if check_processor_modules:
			logger.disable()
			can_process = processor_module.pre_process('stream')
			logger.enable()
		else:
			can_process = True

		if can_process:
			process_inputs =\
			{
				'source_vision_frames': source_vision_frames,
				'source_audio_frame': source_audio_frame,
				'source_voice_frame': source_voice_frame,
				'target_vision_frame': target_vision_frame,
				'temp_vision_frame': temp_vision_frame,
				'temp_vision_mask': temp_vision_mask
			}

			if source_faces and processor_module.__name__ in source_faces:
				process_inputs['source_face'] = source_faces.get(processor_module.__name__)

			temp_vision_frame, temp_vision_mask = processor_module.process_frame(process_inputs)

	return temp_vision_frame


def open_stream(stream_mode : StreamMode, stream_resolution : str, stream_fps : Fps) -> subprocess.Popen[bytes]:
	commands = ffmpeg_builder.chain(
		ffmpeg_builder.capture_video(),
		ffmpeg_builder.set_media_resolution(stream_resolution),
		ffmpeg_builder.set_input_fps(stream_fps)
	)

	if stream_mode == 'udp':
		commands.extend(ffmpeg_builder.set_input('-'))
		commands.extend(ffmpeg_builder.set_stream_mode('udp'))
		commands.extend(ffmpeg_builder.set_stream_quality(2000))
		commands.extend(ffmpeg_builder.set_output('udp://localhost:27000?pkt_size=1316'))

	if stream_mode == 'v4l2':
		device_directory_path = '/sys/devices/virtual/video4linux'
		commands.extend(ffmpeg_builder.set_input('-'))
		commands.extend(ffmpeg_builder.set_stream_mode('v4l2'))

		if is_directory(device_directory_path):
			device_names = os.listdir(device_directory_path)

			for device_name in device_names:
				device_path = '/dev/' + device_name
				commands.extend(ffmpeg_builder.set_output(device_path))

		else:
			logger.error(translator.get('stream_not_loaded').format(stream_mode = stream_mode), __name__)

	return open_ffmpeg(commands)
