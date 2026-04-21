"""
Video / Audio / YouTube → Text transcription app.

Simple Streamlit UI that takes an audio file, a video file, or a YouTube URL
and returns a transcript using OpenAI's Whisper API.

Deploy to Streamlit Cloud:
  1. Push this repo to GitHub (entrypoint: streamlit_app.py).
  2. On share.streamlit.io, set the OPENAI_API_KEY secret under app settings.
  3. `packages.txt` installs ffmpeg on the host.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Video → Text",
    page_icon="🎙️",
    layout="centered",
)

# OpenAI Whisper API hard limit on uploaded file size.
OPENAI_MAX_BYTES = 25 * 1024 * 1024  # 25 MB

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".mp4", ".mpeg", ".mpga", ".webm", ".ogg", ".flac"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpeg", ".mpg", ".wmv", ".flv"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_api_key() -> str | None:
    """Read API key from Streamlit secrets or environment variable."""
    try:
        key = st.secrets.get("OPENAI_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        key = None
    return key or os.environ.get("OPENAI_API_KEY")


def ensure_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def extract_audio(input_path: Path, output_path: Path) -> None:
    """Extract mono 16kHz MP3 audio from any media file using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-b:a", "64k",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-500:]}")


def download_youtube_audio(url: str, output_dir: Path) -> Path:
    """Download audio from a YouTube URL using yt-dlp. Returns path to .mp3."""
    try:
        import yt_dlp  # noqa: F401
    except ImportError as e:
        raise RuntimeError("yt-dlp is not installed.") from e

    from yt_dlp import YoutubeDL

    output_template = str(output_dir / "yt_audio.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "64",
            }
        ],
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    mp3_files = list(output_dir.glob("yt_audio*.mp3"))
    if not mp3_files:
        raise RuntimeError("yt-dlp did not produce an audio file.")
    return mp3_files[0]


def split_audio(input_path: Path, output_dir: Path, chunk_seconds: int = 600) -> list[Path]:
    """Split an audio file into chunks under OPENAI_MAX_BYTES. Returns list of chunk paths."""
    pattern = str(output_dir / "chunk_%03d.mp3")
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-c", "copy",
        pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg segment failed: {result.stderr[-500:]}")
    return sorted(output_dir.glob("chunk_*.mp3"))


def transcribe_file(client: OpenAI, audio_path: Path) -> str:
    """Transcribe a single audio file (must be < 25MB) with Whisper."""
    with audio_path.open("rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text",
        )
    return str(resp).strip()


def transcribe_any(client: OpenAI, audio_path: Path, workdir: Path, progress) -> str:
    """Transcribe an audio file, chunking it if it exceeds the 25MB limit."""
    size = audio_path.stat().st_size
    if size <= OPENAI_MAX_BYTES:
        progress.write(f"Uploading {size / 1_048_576:.1f} MB to Whisper…")
        return transcribe_file(client, audio_path)

    progress.write(
        f"Audio is {size / 1_048_576:.1f} MB — splitting into 10-minute chunks for Whisper…"
    )
    chunk_dir = workdir / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    chunks = split_audio(audio_path, chunk_dir)

    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        progress.write(f"Transcribing chunk {i}/{len(chunks)}…")
        parts.append(transcribe_file(client, chunk))
    return "\n\n".join(p for p in parts if p)


def safe_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return name or "transcript"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🎙️ Video → Text")
st.caption(
    "Upload a video or audio file, or paste a YouTube URL, and get a transcript. "
    "Powered by OpenAI Whisper."
)

api_key = get_api_key()
if not api_key:
    with st.expander("🔑 OpenAI API key required", expanded=True):
        st.warning(
            "No `OPENAI_API_KEY` found in Streamlit secrets or environment. "
            "You can paste a key below for this session, or add it in the app's Secrets on Streamlit Cloud."
        )
        api_key = st.text_input("OpenAI API key", type="password")

if not ensure_ffmpeg():
    st.error(
        "`ffmpeg` is not available on this host. "
        "On Streamlit Cloud, make sure `packages.txt` contains `ffmpeg`."
    )

tab_upload, tab_url = st.tabs(["📁 Upload file", "🔗 YouTube URL"])

with tab_upload:
    uploaded = st.file_uploader(
        "Drop an audio or video file here",
        type=sorted({e.lstrip(".") for e in AUDIO_EXTS | VIDEO_EXTS}),
        accept_multiple_files=False,
    )

with tab_url:
    url = st.text_input("YouTube or other supported URL", placeholder="https://www.youtube.com/watch?v=…")

go = st.button("Transcribe", type="primary", use_container_width=True, disabled=not api_key)

if go:
    if not api_key:
        st.error("Please provide an OpenAI API key.")
        st.stop()

    client = OpenAI(api_key=api_key)
    progress = st.status("Starting…", expanded=True)

    try:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)

            # Resolve input → a local media file path
            if uploaded is not None:
                progress.write(f"Saving uploaded file `{uploaded.name}`…")
                source_name = uploaded.name
                src_path = workdir / safe_filename(uploaded.name)
                src_path.write_bytes(uploaded.getbuffer())
            elif url.strip():
                progress.write("Downloading audio from URL…")
                src_path = download_youtube_audio(url.strip(), workdir)
                source_name = src_path.stem
            else:
                st.error("Upload a file or paste a URL first.")
                st.stop()

            # Always normalize to a small mono 16kHz mp3 before sending to Whisper.
            progress.write("Extracting & compressing audio with ffmpeg…")
            audio_path = workdir / "audio.mp3"
            extract_audio(src_path, audio_path)

            # Transcribe (chunking if needed)
            transcript = transcribe_any(client, audio_path, workdir, progress)

            progress.update(label="Done ✅", state="complete", expanded=False)

        if not transcript:
            st.warning("Whisper returned an empty transcript.")
        else:
            st.subheader("Transcript")
            st.text_area("Transcript", transcript, height=400, label_visibility="collapsed")
            st.download_button(
                "⬇️ Download .txt",
                data=transcript.encode("utf-8"),
                file_name=f"{safe_filename(Path(source_name).stem)}.txt",
                mime="text/plain",
                use_container_width=True,
            )

    except Exception as e:
        progress.update(label="Failed ❌", state="error", expanded=True)
        st.error(f"Something went wrong: {e}")

st.divider()
st.caption(
    "Tip: large videos are auto-converted to compressed mono audio before upload, "
    "and split into 10-minute chunks if they exceed the 25 MB Whisper limit."
)
