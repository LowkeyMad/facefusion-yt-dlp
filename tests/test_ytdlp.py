from unittest.mock import patch

from facefusion.streams.ytdlp import YOUTUBE_STREAM_FORMAT, create_youtube_stream_format, extract_youtube_stream_info, is_youtube_url, resolve_stream_url, resolve_youtube_stream_url


YOUTUBE_STREAM_OUTPUT = '{"url":"https://stream.example.com/live.m3u8","format_id":"95","protocol":"m3u8_native","width":1280,"height":720,"fps":30,"vcodec":"avc1.64001f","acodec":"mp4a.40.2"}'


def test_is_youtube_url() -> None:
	assert is_youtube_url('https://www.youtube.com/watch?v=test') is True
	assert is_youtube_url('https://m.youtube.com/live/test') is True
	assert is_youtube_url('https://youtu.be/test') is True
	assert is_youtube_url('https://example.com/watch?v=test') is False


def test_resolve_stream_url() -> None:
	with patch('facefusion.streams.ytdlp.resolve_youtube_stream_info', return_value = { 'url': 'https://stream.example.com/live.m3u8' }):
		assert resolve_stream_url('https://www.youtube.com/watch?v=test') == 'https://stream.example.com/live.m3u8'

	assert resolve_stream_url('https://stream.example.com/live.m3u8') == 'https://stream.example.com/live.m3u8'


def test_resolve_youtube_stream_url_without_cookies() -> None:
	with patch('os.path.isfile', return_value = False), patch('subprocess.run') as run:
		run.return_value.returncode = 0
		run.return_value.stdout = '\n' + YOUTUBE_STREAM_OUTPUT + '\n'

		assert resolve_youtube_stream_url('https://www.youtube.com/watch?v=test') == 'https://stream.example.com/live.m3u8'
		run.assert_called_once_with([ 'yt-dlp', '--js-runtimes', 'node', '--remote-components', 'ejs:github', '--format', YOUTUBE_STREAM_FORMAT, '--dump-json', 'https://www.youtube.com/watch?v=test' ], capture_output = True, text = True)


def test_resolve_youtube_stream_url_with_cookies() -> None:
	with patch('os.path.isfile', return_value = True), patch('subprocess.run') as run:
		run.return_value.returncode = 0
		run.return_value.stdout = '\n' + YOUTUBE_STREAM_OUTPUT + '\n'

		assert resolve_youtube_stream_url('https://www.youtube.com/watch?v=test') == 'https://stream.example.com/live.m3u8'
		run.assert_called_once_with([ 'yt-dlp', '--cookies', 'cookies.txt', '--js-runtimes', 'node', '--remote-components', 'ejs:github', '--format', YOUTUBE_STREAM_FORMAT, '--dump-json', 'https://www.youtube.com/watch?v=test' ], capture_output = True, text = True)


def test_resolve_youtube_stream_url_with_custom_cookies() -> None:
	with patch('os.path.isfile', return_value = True), patch('subprocess.run') as run:
		run.return_value.returncode = 0
		run.return_value.stdout = '\n' + YOUTUBE_STREAM_OUTPUT + '\n'

		assert resolve_youtube_stream_url('https://www.youtube.com/watch?v=test', '/tmp/facefusion/youtube/cookies.txt') == 'https://stream.example.com/live.m3u8'
		run.assert_called_once_with([ 'yt-dlp', '--cookies', '/tmp/facefusion/youtube/cookies.txt', '--js-runtimes', 'node', '--remote-components', 'ejs:github', '--format', YOUTUBE_STREAM_FORMAT, '--dump-json', 'https://www.youtube.com/watch?v=test' ], capture_output = True, text = True)


def test_resolve_youtube_stream_url_with_selected_quality() -> None:
	with patch('os.path.isfile', return_value = False), patch('subprocess.run') as run:
		run.return_value.returncode = 0
		run.return_value.stdout = '\n{"url":"https://stream.example.com/live-480p.m3u8","format_id":"94","protocol":"m3u8_native","width":854,"height":480,"fps":30,"vcodec":"avc1.4d401f","acodec":"mp4a.40.2"}\n'

		assert resolve_youtube_stream_url('https://www.youtube.com/watch?v=test', None, '480p') == 'https://stream.example.com/live-480p.m3u8'
		run.assert_called_once_with([ 'yt-dlp', '--js-runtimes', 'node', '--remote-components', 'ejs:github', '--format', create_youtube_stream_format('480p'), '--dump-json', 'https://www.youtube.com/watch?v=test' ], capture_output = True, text = True)


def test_create_youtube_stream_format() -> None:
	assert create_youtube_stream_format('auto') == YOUTUBE_STREAM_FORMAT
	assert '[height<=480][protocol^=m3u8]' in create_youtube_stream_format('480p')


def test_extract_youtube_stream_info() -> None:
	stream_info = extract_youtube_stream_info(YOUTUBE_STREAM_OUTPUT)

	assert stream_info.get('url') == 'https://stream.example.com/live.m3u8'
	assert stream_info.get('format_id') == '95'
	assert stream_info.get('protocol') == 'm3u8_native'
	assert stream_info.get('resolution') == '1280x720'
	assert stream_info.get('fps') == 30
	assert stream_info.get('vcodec') == 'avc1.64001f'
	assert stream_info.get('acodec') == 'mp4a.40.2'


def test_resolve_youtube_stream_url_failure() -> None:
	with patch('subprocess.run') as run:
		run.return_value.returncode = 1
		run.return_value.stdout = ''

		assert resolve_youtube_stream_url('https://www.youtube.com/watch?v=test') is None

	with patch('subprocess.run', side_effect = OSError):
		assert resolve_youtube_stream_url('https://www.youtube.com/watch?v=test') is None
