from __future__ import annotations
from pathlib import Path
import yt_dlp
from ftc4.io.audio_utils import ensure_wav

YDL_OPTS = {
    "format": "bestaudio/best",
    "quiet": True,
    "noprogress": True,
    "outtmpl": "%(title)s.%(ext)s",
}


def download_youtube_audio(url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with yt_dlp.YoutubeDL({**YDL_OPTS, "paths": {"home": str(out_dir)}}) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded = Path(out_dir, ydl.prepare_filename(info))
    return ensure_wav(downloaded)
