import os
import subprocess
from typing import Optional
from urllib.parse import urlparse


def is_youtube_url(url : str) -> bool:
	if url:
		netloc = urlparse(url).netloc.lower()
		return netloc in [ 'youtube.com', 'www.youtube.com', 'm.youtube.com', 'youtu.be' ] or netloc.endswith('.youtube.com')
	return False


def resolve_stream_url(url : str) -> Optional[str]:
	if not url:
		return None

	url = url.strip()

	if is_youtube_url(url):
		return resolve_youtube_stream_url(url)

	return url


def resolve_youtube_stream_url(url : str) -> Optional[str]:
	commands = [ 'yt-dlp', '--js-runtimes', 'node', '-g', url ]

	if os.path.isfile('cookies.txt'):
		commands[1:1] = [ '--cookies', 'cookies.txt' ]

	try:
		process = subprocess.run(commands, capture_output = True, text = True)
	except OSError:
		return None

	if process.returncode == 0:
		for stream_url in process.stdout.splitlines():
			if stream_url.strip():
				return stream_url.strip()

	return None
