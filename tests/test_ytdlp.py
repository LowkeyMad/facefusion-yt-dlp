from unittest.mock import patch

from facefusion.streams.ytdlp import is_youtube_url, resolve_stream_url, resolve_youtube_stream_url


def test_is_youtube_url() -> None:
	assert is_youtube_url('https://www.youtube.com/watch?v=test') is True
	assert is_youtube_url('https://m.youtube.com/live/test') is True
	assert is_youtube_url('https://youtu.be/test') is True
	assert is_youtube_url('https://example.com/watch?v=test') is False


def test_resolve_stream_url() -> None:
	with patch('facefusion.streams.ytdlp.resolve_youtube_stream_url', return_value = 'https://stream.example.com/live.m3u8'):
		assert resolve_stream_url('https://www.youtube.com/watch?v=test') == 'https://stream.example.com/live.m3u8'

	assert resolve_stream_url('https://stream.example.com/live.m3u8') == 'https://stream.example.com/live.m3u8'


def test_resolve_youtube_stream_url_without_cookies() -> None:
	with patch('os.path.isfile', return_value = False), patch('subprocess.run') as run:
		run.return_value.returncode = 0
		run.return_value.stdout = '\nhttps://stream.example.com/live.m3u8\n'

		assert resolve_youtube_stream_url('https://www.youtube.com/watch?v=test') == 'https://stream.example.com/live.m3u8'
		run.assert_called_once_with([ 'yt-dlp', '--js-runtimes', 'node', '--remote-components', 'ejs:github', '-g', 'https://www.youtube.com/watch?v=test' ], capture_output = True, text = True)


def test_resolve_youtube_stream_url_with_cookies() -> None:
	with patch('os.path.isfile', return_value = True), patch('subprocess.run') as run:
		run.return_value.returncode = 0
		run.return_value.stdout = '\nhttps://stream.example.com/live.m3u8\n'

		assert resolve_youtube_stream_url('https://www.youtube.com/watch?v=test') == 'https://stream.example.com/live.m3u8'
		run.assert_called_once_with([ 'yt-dlp', '--cookies', 'cookies.txt', '--js-runtimes', 'node', '--remote-components', 'ejs:github', '-g', 'https://www.youtube.com/watch?v=test' ], capture_output = True, text = True)


def test_resolve_youtube_stream_url_with_custom_cookies() -> None:
	with patch('os.path.isfile', return_value = True), patch('subprocess.run') as run:
		run.return_value.returncode = 0
		run.return_value.stdout = '\nhttps://stream.example.com/live.m3u8\n'

		assert resolve_youtube_stream_url('https://www.youtube.com/watch?v=test', '/tmp/facefusion/youtube/cookies.txt') == 'https://stream.example.com/live.m3u8'
		run.assert_called_once_with([ 'yt-dlp', '--cookies', '/tmp/facefusion/youtube/cookies.txt', '--js-runtimes', 'node', '--remote-components', 'ejs:github', '-g', 'https://www.youtube.com/watch?v=test' ], capture_output = True, text = True)


def test_resolve_youtube_stream_url_failure() -> None:
	with patch('subprocess.run') as run:
		run.return_value.returncode = 1
		run.return_value.stdout = ''

		assert resolve_youtube_stream_url('https://www.youtube.com/watch?v=test') is None

	with patch('subprocess.run', side_effect = OSError):
		assert resolve_youtube_stream_url('https://www.youtube.com/watch?v=test') is None
