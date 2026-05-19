from pathlib import Path
from unittest.mock import patch

from facefusion.realtime_diagnostics import RealtimeDiagnostics


class FakeCapture:
	ffmpeg_command = [ 'ffmpeg', '-i', 'https://stream.example.com/live.m3u8' ]

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
	assert 'ffmpeg -i https://stream.example.com/live.m3u8' in report
	assert 'FFmpegCapture read gap stats:' in report
	assert 'Streamer yield/keepalive stats: yields=1' in report
	assert 'Content analyser call count/rate: calls=1' in report
	assert 'Diagnosis:' in report
