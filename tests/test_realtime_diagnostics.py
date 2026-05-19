from pathlib import Path
from unittest.mock import patch

from facefusion.realtime_diagnostics import RealtimeDiagnostics


class FakeCapture:
	ffmpeg_command = [ 'ffmpeg', '-i', 'https://stream.example.com/live.m3u8' ]

	def get_ingest_diagnostics(self) -> dict:
		return\
		{
			'buffering': 'default_probe_analyze',
			'decode_mode': 'cpu',
			'stream_delay': 6.0,
			'buffer_target_frames': 180,
			'buffer_length': 150,
			'dropped_frames': 12,
			'underruns': 2,
			'h264_cuvid_available': True,
			'command_variants':
			{
				'default_probe_analyze': [ 'ffmpeg', '-i', '<stream_url>' ]
			}
		}

	def get_read_gap_stats(self) -> dict:
		return\
		{
			'reads': 3,
			'misses': 1,
			'avg_gap_ms': 33.3,
			'max_gap_ms': 40.0
		}


def test_realtime_diagnostics_writes_report(tmp_path : Path) -> None:
	report_path = tmp_path / 'realtime-diagnostics.log'
	diagnostics = RealtimeDiagnostics(duration = 0, report_path = str(report_path))
	diagnostics.attach_capture(FakeCapture())
	diagnostics.set_stream_quality('480p')
	diagnostics.set_stream_info(
	{
		'format_id': '94',
		'protocol': 'm3u8_native',
		'resolution': '854x480',
		'fps': 30,
		'vcodec': 'avc1.4d401f',
		'acodec': 'mp4a.40.2'
	})
	diagnostics.observe_yield()
	diagnostics.observe_content_analyser()

	def get_state_item(name : str) -> object:
		if name == 'execution_providers':
			return [ 'cuda', 'cpu' ]
		if name == 'processors':
			return [ 'face_swapper' ]
		return None

	with patch('facefusion.realtime_diagnostics.state_manager.get_item', side_effect = get_state_item), patch('facefusion.realtime_diagnostics.onnxruntime.get_available_providers', return_value = [ 'CUDAExecutionProvider', 'CPUExecutionProvider' ]), patch('facefusion.realtime_diagnostics.run_command', return_value = 'nvidia-smi output'):
		diagnostics.finish()

	report = report_path.read_text()

	assert 'Selected FaceFusion execution providers: [\'cuda\', \'cpu\']' in report
	assert 'ONNX Runtime available providers: [\'CUDAExecutionProvider\', \'CPUExecutionProvider\']' in report
	assert 'Active processors: [\'face_swapper\']' in report
	assert 'Face swap enabled: True' in report
	assert 'YouTube stream quality: 480p' in report
	assert 'YouTube selected format: format_id=94, protocol=m3u8_native, resolution=854x480, fps=30, vcodec=avc1.4d401f, acodec=mp4a.40.2' in report
	assert 'FFmpeg decode mode: cpu' in report
	assert 'FFmpeg buffering mode: default_probe_analyze' in report
	assert 'Webcam stream delay: 6.0' in report
	assert 'Webcam stream buffer target frames: 180' in report
	assert 'Webcam stream buffer length: 150' in report
	assert 'Webcam stream dropped frames: 12' in report
	assert 'Webcam stream underruns: 2' in report
	assert 'ffmpeg -i https://stream.example.com/live.m3u8' in report
	assert 'FFmpeg ingest command variants:' in report
	assert 'FFmpegCapture read gap stats:' in report
	assert 'Streamer yield/keepalive stats: yields=1' in report
	assert 'Content analyser call count/rate: calls=1' in report
	assert 'Diagnosis:' in report
