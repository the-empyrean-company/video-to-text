"""
Video / Audio → Text transcription app.

Simple Streamlit UI that takes an audio or video file and returns a transcript
using OpenAI's Whisper API.

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
from openai import (
    OpenAI,
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    RateLimitError,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Qargo Transcribe",
    page_icon="🎙️",
    layout="centered",
)

# Qargo brand colours (from Qargo styling guide).
QARGO_GREEN = "#00E85B"
QARGO_NAVY = "#1C2C31"

# OpenAI Whisper API hard limit on uploaded file size.
OPENAI_MAX_BYTES = 25 * 1024 * 1024  # 25 MB

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".mp4", ".mpeg", ".mpga", ".webm", ".ogg", ".flac"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpeg", ".mpg", ".wmv", ".flv"}


# ---------------------------------------------------------------------------
# Qargo styling
# ---------------------------------------------------------------------------

def inject_qargo_css() -> None:
    """Apply Qargo brand styling on top of the Streamlit theme."""
    st.markdown(
        f"""
        <style>
            /* Headings in Qargo Navy */
            h1, h2, h3, h4 {{
                color: {QARGO_NAVY} !important;
                font-weight: 600;
            }}

            /* Accent bar above the main title */
            .qargo-accent {{
                display: inline-block;
                width: 56px;
                height: 6px;
                background: {QARGO_GREEN};
                border-radius: 3px;
                margin-bottom: 12px;
            }}

            /* Primary button: Qargo Green fill, Navy text */
            .stButton > button[kind="primary"] {{
                background-color: {QARGO_GREEN};
                color: {QARGO_NAVY};
                border: 0;
                border-radius: 10px;
                font-weight: 600;
                padding: 0.6rem 1.2rem;
                transition: filter 0.15s ease-in-out;
            }}
            .stButton > button[kind="primary"]:hover {{
                filter: brightness(0.92);
                color: {QARGO_NAVY};
            }}
            .stButton > button[kind="primary"]:disabled {{
                background-color: #E5E7EA;
                color: #9AA3A7;
            }}

            /* File uploader: soften borders, add Qargo accent */
            [data-testid="stFileUploader"] section {{
                border: 1.5px dashed rgba(28, 44, 49, 0.25);
                border-radius: 12px;
                background: #FAFBFC;
            }}
            [data-testid="stFileUploader"] section:hover {{
                border-color: {QARGO_GREEN};
            }}

            /* Download button: outlined Navy */
            .stDownloadButton > button {{
                background: white;
                color: {QARGO_NAVY};
                border: 1.5px solid {QARGO_NAVY};
                border-radius: 10px;
                font-weight: 600;
            }}
            .stDownloadButton > button:hover {{
                background: {QARGO_NAVY};
                color: white;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def is_insufficient_quota(err: Exception) -> bool:
    """Return True if the OpenAI error indicates credits are exhausted."""
    code = getattr(err, "code", None)
    if code == "insufficient_quota":
        return True
    # Fallback: inspect the body returned by the API.
    body = getattr(err, "body", None)
    if isinstance(body, dict):
        inner = body.get("error") if isinstance(body.get("error"), dict) else body
        if isinstance(inner, dict) and inner.get("code") == "insufficient_quota":
            return True
    msg = str(err).lower()
    return "insufficient_quota" in msg or "exceeded your current quota" in msg


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

inject_qargo_css()

st.markdown('<div class="qargo-accent"></div>', unsafe_allow_html=True)
st.title("Qargo Transcribe")
st.caption("Upload a video or audio file and get a plain-text transcript. Powered by OpenAI Whisper.")

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

uploaded = st.file_uploader(
    "Drop an audio or video file here",
    type=sorted({e.lstrip(".") for e in AUDIO_EXTS | VIDEO_EXTS}),
    accept_multiple_files=False,
)

go = st.button("Transcribe", type="primary", use_container_width=True, disabled=not api_key)

if go:
    if not api_key:
        st.error("Please provide an OpenAI API key.")
        st.stop()

    if uploaded is None:
        st.error("Upload a file first.")
        st.stop()

    client = OpenAI(api_key=api_key)
    progress = st.status("Starting…", expanded=True)
    source_name = uploaded.name
    transcript: str | None = None

    try:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)

            progress.write(f"Saving uploaded file `{uploaded.name}`…")
            src_path = workdir / safe_filename(uploaded.name)
            src_path.write_bytes(uploaded.getbuffer())

            # Always normalize to a small mono 16kHz mp3 before sending to Whisper.
            progress.write("Extracting & compressing audio with ffmpeg…")
            audio_path = workdir / "audio.mp3"
            extract_audio(src_path, audio_path)

            # Transcribe (chunking if needed)
            transcript = transcribe_any(client, audio_path, workdir, progress)

            progress.update(label="Done ✅", state="complete", expanded=False)

    except RateLimitError as e:
        progress.update(label="Stopped ❌", state="error", expanded=True)
        if is_insufficient_quota(e):
            st.error(
                "💳 **Your OpenAI credits are exhausted.** "
                "Add credits at https://platform.openai.com/account/billing to continue transcribing."
            )
        else:
            st.error(
                "⚠️ Hit a rate limit from OpenAI. Please wait a moment and try again."
            )
    except AuthenticationError:
        progress.update(label="Stopped ❌", state="error", expanded=True)
        st.error("🔑 The OpenAI API key was rejected. Check that the key is correct and active.")
    except APIConnectionError:
        progress.update(label="Stopped ❌", state="error", expanded=True)
        st.error("🌐 Could not reach OpenAI. Check your connection and try again.")
    except APIStatusError as e:
        progress.update(label="Stopped ❌", state="error", expanded=True)
        # Catch billing-related errors that come through as plain 4xx status errors too.
        if is_insufficient_quota(e):
            st.error(
                "💳 **Your OpenAI credits are exhausted.** "
                "Add credits at https://platform.openai.com/account/billing to continue transcribing."
            )
        else:
            st.error(f"OpenAI returned an error: {e}")
    except Exception as e:
        progress.update(label="Failed ❌", state="error", expanded=True)
        st.error(f"Something went wrong: {e}")
    else:
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

st.divider()
st.caption(
    "Large videos are auto-converted to compressed mono audio before upload, "
    "and split into 10-minute chunks if they exceed the 25 MB Whisper limit."
)
