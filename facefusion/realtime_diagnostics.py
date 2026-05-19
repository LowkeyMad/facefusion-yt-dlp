import os
import shutil
import subprocess
from statistics import mean
from time import monotonic
from typing import Any, Dict, List, Optional

import onnxruntime

from facefusion import state_manager

REPORT_PATH = 'realtime-diagnostics.log'
DEFAULT_DURATION = 60.0


class RealtimeDiagnostics:
	def __init__(self, duration : float = DEFAULT_DURATION, report_path : str = REPORT_PATH) -> None:
		self.duration = duration
		self.report_path = report_path
		self.started_at = monotonic()
		self.finished = False
		self.python_pid = os.getpid()
		self.ffmpeg_pids : List[int] = []
		self.ffmpeg_commands : List[List[str]] = []
		self.process_samples : Dict[str, List[Dict[str, Optional[float]]]] =\
		{
			'python': [],
			'ffmpeg': []
		}
		self.last_sample_at = 0.0
		self.streamer_yields = 0
		self.streamer_keepalives = 0
		self.content_analyser_calls = 0
		self.content_analyser_started_at : Optional[float] = None
		self.capture : Optional[Any] = None
		self.stream_quality : Optional[str] = None
		self.stream_info : Dict[str, Any] = {}
		self.ingest_diagnostics : Dict[str, Any] = {}

	def attach_capture(self, camera_capture : Any) -> None:
		self.capture = camera_capture
		self._attach_process(camera_capture)
		if hasattr(camera_capture, 'get_ingest_diagnostics'):
			self.ingest_diagnostics = camera_capture.get_ingest_diagnostics()

	def attach_stream(self, stream : Any) -> None:
		self._attach_process(stream)

	def set_stream_quality(self, stream_quality : str) -> None:
		self.stream_quality = stream_quality

	def set_stream_info(self, stream_info : Dict[str, Any]) -> None:
		self.stream_info = stream_info

	def observe_keepalive(self) -> None:
		self.streamer_keepalives += 1
		self.sample_processes()

	def observe_yield(self) -> None:
		self.streamer_yields += 1
		self.sample_processes()

	def observe_content_analyser(self) -> None:
		if self.content_analyser_started_at is None:
			self.content_analyser_started_at = monotonic()
		self.content_analyser_calls += 1
		self.sample_processes()

	def sample_processes(self) -> None:
		now = monotonic()

		if now - self.last_sample_at < 1.0:
			return

		self.last_sample_at = now
		self.process_samples['python'].append(create_process_sample(self.python_pid))

		for ffmpeg_pid in self.ffmpeg_pids:
			self.process_samples['ffmpeg'].append(create_process_sample(ffmpeg_pid))

	def should_finish(self) -> bool:
		return not self.finished and monotonic() - self.started_at >= self.duration

	def finish(self) -> None:
		if self.finished:
			return

		self.finished = True
		if self.capture and hasattr(self.capture, 'get_ingest_diagnostics'):
			self.ingest_diagnostics = self.capture.get_ingest_diagnostics()
		self.sample_processes()
		write_report(self.report_path, self.create_report())

	def create_report(self) -> str:
		elapsed = max(monotonic() - self.started_at, 0.001)
		read_gap_stats = collect_read_gap_stats(self.capture)
		content_analyser_rate = self.content_analyser_calls / elapsed
		lines =\
		[
			'FaceFusion realtime diagnostics',
			'',
			'Selected FaceFusion execution providers: ' + str(state_manager.get_item('execution_providers')),
			'ONNX Runtime available providers: ' + str(onnxruntime.get_available_providers()),
			'Active processors: ' + str(state_manager.get_item('processors')),
			'Face swap enabled: ' + str('face_swapper' in (state_manager.get_item('processors') or [])),
			'YouTube stream quality: ' + str(self.stream_quality or 'auto'),
			'YouTube selected format: ' + format_stream_info(self.stream_info),
			'FFmpeg decode mode: ' + str(self.ingest_diagnostics.get('decode_mode') or 'unknown'),
			'FFmpeg buffering mode: ' + str(self.ingest_diagnostics.get('buffering') or 'unknown'),
			'Webcam stream delay: ' + str(self.ingest_diagnostics.get('stream_delay')),
			'Webcam stream buffer target frames: ' + str(self.ingest_diagnostics.get('buffer_target_frames')),
			'Webcam stream buffer length: ' + str(self.ingest_diagnostics.get('buffer_length')),
			'Webcam stream dropped frames: ' + str(self.ingest_diagnostics.get('dropped_frames')),
			'Webcam stream underruns: ' + str(self.ingest_diagnostics.get('underruns')),
			'FFmpeg h264_cuvid available: ' + str(self.ingest_diagnostics.get('h264_cuvid_available')),
			'',
			'FFmpeg commands:'
		]

		if self.ffmpeg_commands:
			for ffmpeg_command in self.ffmpeg_commands:
				lines.append('  ' + join_command(ffmpeg_command))
		else:
			lines.append('  none captured')

		lines.extend(
		[
			'',
			'Python process CPU/RSS: ' + summarize_process_samples(self.process_samples['python']),
			'FFmpeg process CPU/RSS: ' + summarize_process_samples(self.process_samples['ffmpeg']),
			'',
			'nvidia-smi summary:',
			indent_text(run_command([ 'nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,utilization.memory,temperature.gpu', '--format=csv,noheader,nounits' ])),
			'',
			'nvidia-smi pmon sample:',
			indent_text(run_command([ 'nvidia-smi', 'pmon', '-c', '1' ])),
			'',
			'FFmpeg ingest command variants:',
			indent_text(format_command_variants(self.ingest_diagnostics.get('command_variants') or {})),
			'',
			'FFmpegCapture read gap stats: ' + str(read_gap_stats),
			'Streamer yield/keepalive stats: yields=' + str(self.streamer_yields) + ', keepalives=' + str(self.streamer_keepalives) + ', yield_rate=' + format_rate(self.streamer_yields / elapsed) + '/s',
			'Content analyser call count/rate: calls=' + str(self.content_analyser_calls) + ', rate=' + format_rate(content_analyser_rate) + '/s',
			'',
			'Diagnosis: ' + diagnose(read_gap_stats, self.streamer_yields, self.streamer_keepalives, content_analyser_rate, self.process_samples)
		])

		return os.linesep.join(lines) + os.linesep

	def _attach_process(self, item : Any) -> None:
		process = getattr(item, 'process', item)
		pid = getattr(process, 'pid', None)

		if isinstance(pid, int) and pid not in self.ffmpeg_pids:
			self.ffmpeg_pids.append(pid)

		command = getattr(item, 'ffmpeg_command', None) or getattr(process, 'facefusion_command', None)

		if command and command not in self.ffmpeg_commands:
			self.ffmpeg_commands.append(command)


def create_process_sample(pid : int) -> Dict[str, Optional[float]]:
	try:
		with open('/proc/' + str(pid) + '/stat') as stat_file:
			stat_values = stat_file.read().split()

		clock_ticks = os.sysconf(os.sysconf_names['SC_CLK_TCK'])
		page_size = os.sysconf(os.sysconf_names['SC_PAGE_SIZE'])
		return\
		{
			'time': monotonic(),
			'cpu_time': (float(stat_values[13]) + float(stat_values[14])) / clock_ticks,
			'rss': float(stat_values[23]) * page_size / 1024 / 1024
		}
	except Exception:
		return\
		{
			'time': monotonic(),
			'cpu_time': None,
			'rss': None
		}


def summarize_process_samples(samples : List[Dict[str, Optional[float]]]) -> str:
	rss_values = [ sample.get('rss') for sample in samples if sample.get('rss') is not None ]
	cpu_percent = calculate_cpu_percent(samples)

	if not samples:
		return 'no samples'

	return 'samples=' + str(len(samples)) + ', avg_cpu=' + format_rate(cpu_percent) + '%, avg_rss=' + format_rate(mean(rss_values) if rss_values else None) + ' MiB'


def calculate_cpu_percent(samples : List[Dict[str, Optional[float]]]) -> Optional[float]:
	valid_samples = [ sample for sample in samples if sample.get('cpu_time') is not None ]

	if len(valid_samples) < 2:
		return None

	time_delta = valid_samples[-1]['time'] - valid_samples[0]['time'] #type:ignore[operator]
	cpu_delta = valid_samples[-1]['cpu_time'] - valid_samples[0]['cpu_time'] #type:ignore[operator]

	if time_delta <= 0:
		return None

	return 100.0 * cpu_delta / time_delta


def collect_read_gap_stats(capture : Optional[Any]) -> Dict[str, Optional[float]]:
	if capture and hasattr(capture, 'get_read_gap_stats'):
		return capture.get_read_gap_stats()

	return\
	{
		'reads': None,
		'misses': None,
		'avg_gap_ms': None,
		'max_gap_ms': None
	}


def run_command(command : List[str]) -> str:
	if not shutil.which(command[0]):
		return command[0] + ' not found'

	try:
		return subprocess.check_output(command, stderr = subprocess.STDOUT, timeout = 5).decode(errors = 'replace').strip() or 'no output'
	except Exception as exception:
		return type(exception).__name__ + ': ' + str(exception)


def diagnose(read_gap_stats : Dict[str, Optional[float]], streamer_yields : int, streamer_keepalives : int, content_analyser_rate : float, process_samples : Dict[str, List[Dict[str, Optional[float]]]]) -> str:
	python_cpu = calculate_cpu_percent(process_samples.get('python', []))
	ffmpeg_cpu = calculate_cpu_percent(process_samples.get('ffmpeg', []))
	max_gap_ms = read_gap_stats.get('max_gap_ms') or 0

	if streamer_yields == 0:
		return 'No frames were yielded during the diagnostic window; check capture connectivity, FFmpeg decode, and stream URL/device availability.'
	if max_gap_ms > 500:
		return 'Capture read gaps are high, which points to input decode/network stalls before FaceFusion processing.'
	if python_cpu and python_cpu > 85:
		return 'Python CPU usage is high; the bottleneck is likely frame processing or content analysis on the selected execution providers.'
	if ffmpeg_cpu and ffmpeg_cpu > 85:
		return 'FFmpeg CPU usage is high; decode, scaling, or output streaming is likely limiting throughput.'
	if streamer_keepalives > streamer_yields:
		return 'The streamer spent more time waiting than yielding frames; capture delivery is likely intermittent.'
	if content_analyser_rate > 0 and content_analyser_rate < 1:
		return 'Content analyser calls are sparse; frame delivery or upstream processing is likely slower than expected.'

	return 'No single bottleneck dominates the captured evidence; compare frame rate, GPU utilization, and processor settings for the remaining throughput limit.'


def write_report(report_path : str, report : str) -> None:
	with open(report_path, 'w') as report_file:
		report_file.write(report)


def indent_text(text : str) -> str:
	return os.linesep.join('  ' + line for line in text.splitlines())


def format_stream_info(stream_info : Dict[str, Any]) -> str:
	if not stream_info:
		return 'none'

	return ', '.join(
	[
		'format_id=' + str(stream_info.get('format_id')),
		'protocol=' + str(stream_info.get('protocol')),
		'resolution=' + str(stream_info.get('resolution')),
		'fps=' + str(stream_info.get('fps')),
		'vcodec=' + str(stream_info.get('vcodec')),
		'acodec=' + str(stream_info.get('acodec'))
	])


def format_command_variants(command_variants : Dict[str, List[str]]) -> str:
	if not command_variants:
		return 'none'

	return os.linesep.join(variant_name + ': ' + join_command(command) for variant_name, command in command_variants.items())


def join_command(command : List[str]) -> str:
	return ' '.join(str(command_part) for command_part in command)


def format_rate(value : Optional[float]) -> str:
	if value is None:
		return 'n/a'

	return format(value, '.2f')
