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

# Character-based chunk size for the GPT formatter. gpt-4o-mini has a 128k-token
# context window but a ~16k-token *output* cap; at roughly 4 chars per token, a
# 24k-char input chunk produces well under the output limit with headroom for
# slight expansion (punctuation, paragraph breaks).
FORMAT_CHUNK_CHARS = 24_000

# System prompt for the transcript formatter. Kept at module scope so it's not
# re-allocated on every call / chunk.
FORMAT_SYSTEM_PROMPT = (
    "You are an editor that cleans up raw speech-to-text transcripts.\n"
    "Your job is to:\n"
    "- Add proper punctuation and capitalization.\n"
    "- Break the text into readable paragraphs.\n"
    "- Fix obvious transcription errors (wrong homophones, misheard words) "
    "using the surrounding context.\n"
    "- Remove filler words ('um', 'uh', 'you know', 'like') when they add no meaning.\n"
    "- Preserve the speaker's original meaning and wording wherever possible — "
    "do NOT paraphrase or summarise.\n"
    "- Do not add content that wasn't in the transcript.\n"
    "- Keep the transcript in its original language.\n"
    "- The input may be one segment of a longer transcript, so it can start or "
    "end mid-thought — clean it up as-is without inventing context.\n"
    "- Respond with ONLY the cleaned transcript, no preamble or commentary."
)

# Split on sentence-ending punctuation followed by whitespace.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")

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


def split_for_format(text: str, chunk_chars: int = FORMAT_CHUNK_CHARS) -> list[str]:
    """Split a transcript into chunks that fit in one format call.

    Prefers sentence boundaries (so chunks don't end mid-sentence), falls back
    to whitespace, and finally hard-splits — so no content is ever dropped.
    """
    text = text.strip()
    if len(text) <= chunk_chars:
        return [text] if text else []

    sentences = _SENTENCE_SPLIT_RE.split(text)
    chunks: list[str] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        if buf:
            chunks.append(buf)
            buf = ""

    for sent in sentences:
        if not sent:
            continue
        candidate = f"{buf} {sent}".strip() if buf else sent
        if len(candidate) <= chunk_chars:
            buf = candidate
            continue
        # Adding this sentence would overflow the chunk — flush what we have.
        flush()
        # A single sentence could itself be longer than the chunk limit
        # (e.g., Whisper output with no sentence-ending punctuation).
        if len(sent) > chunk_chars:
            words = sent.split()
            piece = ""
            for word in words:
                cand = f"{piece} {word}".strip() if piece else word
                if len(cand) <= chunk_chars:
                    piece = cand
                else:
                    if piece:
                        chunks.append(piece)
                    # Final safety net: hard-split words that exceed the limit
                    # (shouldn't happen with real speech, but don't crash).
                    while len(word) > chunk_chars:
                        chunks.append(word[:chunk_chars])
                        word = word[chunk_chars:]
                    piece = word
            buf = piece
        else:
            buf = sent
    flush()
    return chunks


def format_transcript(
    client: OpenAI,
    transcript: str,
    *,
    model: str = "gpt-4o-mini",
    progress=None,
) -> str:
    """Clean up a raw Whisper transcript, chunking it if it's too long for one call."""
    chunks = split_for_format(transcript)
    if not chunks:
        return ""

    cleaned_parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        if progress is not None:
            if len(chunks) == 1:
                progress.write(
                    f"Sending {len(chunk):,} characters to {model} for cleanup…"
                )
            else:
                progress.write(
                    f"Formatting chunk {i}/{len(chunks)} "
                    f"({len(chunk):,} characters)…"
                )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": FORMAT_SYSTEM_PROMPT},
                {"role": "user", "content": chunk},
            ],
            temperature=0.2,
        )
        cleaned_parts.append((resp.choices[0].message.content or "").strip())

    return "\n\n".join(p for p in cleaned_parts if p)


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

# Session state: keep transcripts around across Streamlit reruns so the
# "Format & correct" button still has something to work on after the
# transcription run ends.
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "formatted_transcript" not in st.session_state:
    st.session_state.formatted_transcript = None
if "source_name" not in st.session_state:
    st.session_state.source_name = None

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
            st.session_state.transcript = None
            st.session_state.formatted_transcript = None
            st.session_state.source_name = None
        else:
            # Persist across reruns so the "Format & correct" button can use it.
            st.session_state.transcript = transcript
            st.session_state.source_name = source_name
            st.session_state.formatted_transcript = None

# ---------------------------------------------------------------------------
# Transcript display + optional GPT formatting
# ---------------------------------------------------------------------------

if st.session_state.transcript:
    transcript = st.session_state.transcript
    source_name = st.session_state.source_name or "transcript"
    stem = safe_filename(Path(source_name).stem)

    st.subheader("Transcript")
    st.text_area(
        "Transcript",
        transcript,
        height=400,
        label_visibility="collapsed",
        key="raw_transcript_display",
    )

    col_dl, col_fmt = st.columns(2)
    with col_dl:
        st.download_button(
            "⬇️ Download .txt",
            data=transcript.encode("utf-8"),
            file_name=f"{stem}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with col_fmt:
        format_clicked = st.button(
            "✨ Format & correct with GPT",
            use_container_width=True,
            disabled=not api_key,
            help=(
                "Sends the transcript to GPT to add punctuation, fix obvious "
                "transcription errors, and break it into readable paragraphs."
            ),
        )

    if format_clicked:
        if not api_key:
            st.error("Please provide an OpenAI API key.")
        else:
            fmt_client = OpenAI(api_key=api_key)
            fmt_progress = st.status("Formatting with GPT…", expanded=True)
            try:
                cleaned = format_transcript(
                    fmt_client, transcript, progress=fmt_progress
                )
                st.session_state.formatted_transcript = cleaned
                fmt_progress.update(
                    label="Formatted ✅", state="complete", expanded=False
                )
            except RateLimitError as e:
                fmt_progress.update(label="Stopped ❌", state="error", expanded=True)
                if is_insufficient_quota(e):
                    st.error(
                        "💳 **Your OpenAI credits are exhausted.** "
                        "Add credits at https://platform.openai.com/account/billing to continue."
                    )
                else:
                    st.error("⚠️ Hit a rate limit from OpenAI. Please wait a moment and try again.")
            except AuthenticationError:
                fmt_progress.update(label="Stopped ❌", state="error", expanded=True)
                st.error("🔑 The OpenAI API key was rejected. Check that the key is correct and active.")
            except APIConnectionError:
                fmt_progress.update(label="Stopped ❌", state="error", expanded=True)
                st.error("🌐 Could not reach OpenAI. Check your connection and try again.")
            except APIStatusError as e:
                fmt_progress.update(label="Stopped ❌", state="error", expanded=True)
                if is_insufficient_quota(e):
                    st.error(
                        "💳 **Your OpenAI credits are exhausted.** "
                        "Add credits at https://platform.openai.com/account/billing to continue."
                    )
                else:
                    st.error(f"OpenAI returned an error: {e}")
            except Exception as e:
                fmt_progress.update(label="Failed ❌", state="error", expanded=True)
                st.error(f"Something went wrong while formatting: {e}")

    if st.session_state.formatted_transcript:
        formatted = st.session_state.formatted_transcript
        st.subheader("Formatted transcript")
        st.text_area(
            "Formatted transcript",
            formatted,
            height=400,
            label_visibility="collapsed",
            key="formatted_transcript_display",
        )
        st.download_button(
            "⬇️ Download formatted .txt",
            data=formatted.encode("utf-8"),
            file_name=f"{stem}_formatted.txt",
            mime="text/plain",
            use_container_width=True,
            key="download_formatted",
        )

st.divider()
st.caption(
    "There's an upload limit of 2GB, contact Álvaro for larger files."
)
