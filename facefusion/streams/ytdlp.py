import json
import os
import subprocess
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from facefusion.types import WebcamStreamQuality

YOUTUBE_STREAM_FORMAT = 'best[vcodec^=avc1]/best[ext=mp4]/best'
YOUTUBE_STREAM_QUALITY_HEIGHTS =\
{
	'1080p': 1080,
	'720p': 720,
	'480p': 480,
	'360p': 360
}


def is_youtube_url(url : str) -> bool:
	if url:
		netloc = urlparse(url).netloc.lower()
		return netloc in [ 'youtube.com', 'www.youtube.com', 'm.youtube.com', 'youtu.be' ] or netloc.endswith('.youtube.com')
	return False


def resolve_stream_url(url : str, cookies_path : Optional[str] = None, stream_quality : WebcamStreamQuality = 'auto') -> Optional[str]:
	stream_info = resolve_stream_info(url, cookies_path, stream_quality)

	if stream_info:
		return stream_info.get('url')

	return None


def resolve_stream_info(url : str, cookies_path : Optional[str] = None, stream_quality : WebcamStreamQuality = 'auto') -> Optional[Dict[str, Any]]:
	if not url:
		return None

	url = url.strip()

	if is_youtube_url(url):
		return resolve_youtube_stream_info(url, cookies_path, stream_quality)

	return\
	{
		'url': url
	}


def resolve_youtube_stream_url(url : str, cookies_path : Optional[str] = None, stream_quality : WebcamStreamQuality = 'auto') -> Optional[str]:
	stream_info = resolve_youtube_stream_info(url, cookies_path, stream_quality)

	if stream_info:
		return stream_info.get('url')

	return None


def resolve_youtube_stream_info(url : str, cookies_path : Optional[str] = None, stream_quality : WebcamStreamQuality = 'auto') -> Optional[Dict[str, Any]]:
	commands = [ 'yt-dlp', '--js-runtimes', 'node', '--remote-components', 'ejs:github', '--format', create_youtube_stream_format(stream_quality), '--dump-json', url ]

	if cookies_path and os.path.isfile(cookies_path):
		commands[1:1] = [ '--cookies', cookies_path ]
	elif os.path.isfile('cookies.txt'):
		commands[1:1] = [ '--cookies', 'cookies.txt' ]

	try:
		process = subprocess.run(commands, capture_output = True, text = True)
	except OSError:
		return None

	if process.returncode == 0:
		return extract_youtube_stream_info(process.stdout)

	return None


def create_youtube_stream_format(stream_quality : WebcamStreamQuality) -> str:
	height = YOUTUBE_STREAM_QUALITY_HEIGHTS.get(stream_quality)

	if not height:
		return YOUTUBE_STREAM_FORMAT

	return '/'.join(
	[
		'best[vcodec^=avc1][height<=' + str(height) + '][protocol^=m3u8]',
		'best[height<=' + str(height) + '][protocol^=m3u8]',
		'best[vcodec^=avc1][height<=' + str(height) + ']',
		'best[height<=' + str(height) + ']',
		YOUTUBE_STREAM_FORMAT
	])


def extract_youtube_stream_info(output : str) -> Optional[Dict[str, Any]]:
	for line in reversed(output.splitlines()):
		if line.strip():
			try:
				payload = json.loads(line)
			except json.JSONDecodeError:
				return None

			format_payload = select_video_format_payload(payload)
			stream_url = format_payload.get('url') or payload.get('url')

			if stream_url:
				width = format_payload.get('width') or payload.get('width')
				height = format_payload.get('height') or payload.get('height')
				return\
				{
					'url': stream_url,
					'format_id': format_payload.get('format_id') or payload.get('format_id'),
					'protocol': format_payload.get('protocol') or payload.get('protocol'),
					'resolution': format_payload.get('resolution') or payload.get('resolution') or create_resolution_text(width, height),
					'fps': format_payload.get('fps') or payload.get('fps'),
					'vcodec': format_payload.get('vcodec') or payload.get('vcodec'),
					'acodec': format_payload.get('acodec') or payload.get('acodec')
				}

	return None


def select_video_format_payload(payload : Dict[str, Any]) -> Dict[str, Any]:
	for format_payload in payload.get('requested_formats') or []:
		if format_payload.get('vcodec') != 'none' and format_payload.get('url'):
			return format_payload

	return payload


def create_resolution_text(width : Optional[int], height : Optional[int]) -> Optional[str]:
	if width and height:
		return str(width) + 'x' + str(height)

	return None
