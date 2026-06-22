#!/usr/bin/env python3
"""
Qargo Transcribe — command-line edition.

A drag-and-drop friendly transcriber. Turns audio/video files into plain-text
transcripts with OpenAI Whisper and drops the .txt straight onto your Desktop.
No file-size limit other than what your own machine can handle.

------------------------------------------------------------------------------
QUICK START (the easy way)
------------------------------------------------------------------------------
  1. Open Terminal.
  2. Type:  python3 transcribe_cli.py      (then press Enter)
  3. When prompted, DRAG your file(s) from Finder into the Terminal window and
     press Enter. You can drop several at once.
  4. The transcript(s) appear on your Desktop. Done.

------------------------------------------------------------------------------
ALSO WORKS
------------------------------------------------------------------------------
  python3 transcribe_cli.py video.mp4 "/path/with spaces/meeting.m4a"
  python3 transcribe_cli.py --format interview.mov     # also GPT-clean it
  python3 transcribe_cli.py --outdir ~/Documents clip.mp4

------------------------------------------------------------------------------
ONE-TIME SETUP
------------------------------------------------------------------------------
  - Install ffmpeg:        macOS:  brew install ffmpeg
  - Install the library:   pip3 install openai
  - Set your OpenAI key:   export OPENAI_API_KEY="sk-..."
    (or just paste it when the script asks).
"""

from __future__ import annotations

import argparse
import getpass
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    sys.exit(
        "The 'openai' package is not installed.\n"
        "Fix it with:  pip3 install openai"
    )

# --- Settings ---------------------------------------------------------------

OPENAI_MAX_BYTES = 25 * 1024 * 1024  # Whisper's hard 25 MB per-file limit.
FORMAT_CHUNK_CHARS = 24_000
WHISPER_MODEL = "whisper-1"
FORMAT_MODEL = "gpt-4o-mini"

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".mp4", ".mpeg", ".mpga", ".webm", ".ogg", ".flac"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpeg", ".mpg", ".wmv", ".flv"}
MEDIA_EXTS = AUDIO_EXTS | VIDEO_EXTS

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

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")

# --- Tiny console helpers ---------------------------------------------------

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def info(msg: str) -> None:
    print(f"  {msg}")


def step(msg: str) -> None:
    print(_c(f"→ {msg}", "36"))  # cyan


def ok(msg: str) -> None:
    print(_c(f"✓ {msg}", "32"))  # green


def warn(msg: str) -> None:
    print(_c(f"! {msg}", "33"))  # yellow


def err(msg: str) -> None:
    print(_c(f"✗ {msg}", "31"))  # red


# --- Core transcription logic (shared with the Streamlit app) ---------------

def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        err("ffmpeg is not installed or not on your PATH.")
        info("Install it with:  brew install ffmpeg   (macOS)")
        info("                  sudo apt install ffmpeg   (Ubuntu/Debian)")
        sys.exit(1)


def get_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key.strip()
    warn("No OPENAI_API_KEY found in your environment.")
    try:
        key = getpass.getpass("Paste your OpenAI API key (it stays hidden): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit("No API key provided.")
    if not key:
        sys.exit("No API key provided.")
    return key


def extract_audio(input_path: Path, output_path: Path) -> None:
    """Extract mono 16 kHz MP3 audio from any media file using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-500:]}")


def split_audio(input_path: Path, output_dir: Path, chunk_seconds: int = 600) -> list[Path]:
    """Split audio into ~10-minute chunks so each stays under Whisper's 25 MB cap."""
    pattern = str(output_dir / "chunk_%03d.mp3")
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-f", "segment", "-segment_time", str(chunk_seconds), "-c", "copy",
        pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg segment failed: {result.stderr[-500:]}")
    return sorted(output_dir.glob("chunk_*.mp3"))


def transcribe_file(client: OpenAI, audio_path: Path) -> str:
    with audio_path.open("rb") as f:
        resp = client.audio.transcriptions.create(
            model=WHISPER_MODEL, file=f, response_format="text",
        )
    return str(resp).strip()


def transcribe_any(client: OpenAI, audio_path: Path, workdir: Path) -> str:
    size = audio_path.stat().st_size
    if size <= OPENAI_MAX_BYTES:
        info(f"Uploading {size / 1_048_576:.1f} MB to Whisper…")
        return transcribe_file(client, audio_path)

    info(f"Audio is {size / 1_048_576:.1f} MB — splitting into 10-minute chunks…")
    chunk_dir = workdir / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    chunks = split_audio(audio_path, chunk_dir)
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        info(f"Transcribing chunk {i}/{len(chunks)}…")
        parts.append(transcribe_file(client, chunk))
    return "\n\n".join(p for p in parts if p)


def split_for_format(text: str, chunk_chars: int = FORMAT_CHUNK_CHARS) -> list[str]:
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
        flush()
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
                    while len(word) > chunk_chars:
                        chunks.append(word[:chunk_chars])
                        word = word[chunk_chars:]
                    piece = word
            buf = piece
        else:
            buf = sent
    flush()
    return chunks


def format_transcript(client: OpenAI, transcript: str) -> str:
    chunks = split_for_format(transcript)
    if not chunks:
        return ""
    cleaned: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        if len(chunks) > 1:
            info(f"Cleaning chunk {i}/{len(chunks)} with {FORMAT_MODEL}…")
        else:
            info(f"Cleaning transcript with {FORMAT_MODEL}…")
        resp = client.chat.completions.create(
            model=FORMAT_MODEL,
            messages=[
                {"role": "system", "content": FORMAT_SYSTEM_PROMPT},
                {"role": "user", "content": chunk},
            ],
            temperature=0.2,
        )
        cleaned.append((resp.choices[0].message.content or "").strip())
    return "\n\n".join(p for p in cleaned if p)


# --- File / path handling ---------------------------------------------------

def safe_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip("_ ")
    return name or "transcript"


def default_output_dir() -> Path:
    """The user's Desktop, or home if there's no Desktop folder."""
    desktop = Path.home() / "Desktop"
    if desktop.is_dir():
        return desktop
    return Path.home()


def unique_path(directory: Path, stem: str, suffix: str = ".txt") -> Path:
    """Return a non-clobbering path: 'name.txt', then 'name (2).txt', etc."""
    candidate = directory / f"{stem}{suffix}"
    n = 2
    while candidate.exists():
        candidate = directory / f"{stem} ({n}){suffix}"
        n += 1
    return candidate


def parse_dropped_line(line: str) -> list[Path]:
    """Turn a line of dragged-in paths into a list of Paths.

    Terminal drag-and-drop pastes paths separated by spaces, with spaces inside
    a path backslash-escaped or the whole path quoted. shlex handles both.
    """
    line = line.strip()
    if not line:
        return []
    try:
        tokens = shlex.split(line)
    except ValueError:
        # Unbalanced quotes — fall back to a naive split.
        tokens = line.split()
    return [Path(os.path.expanduser(t)) for t in tokens if t]


def collect_paths_interactively() -> list[Path]:
    print()
    print(_c("Drag your audio/video file(s) into this window, then press Enter.", "1"))
    info("(You can drop several at once. Just press Enter on an empty line to cancel.)")
    try:
        line = input("\n  Files: ")
    except (EOFError, KeyboardInterrupt):
        print()
        return []
    return parse_dropped_line(line)


# --- Per-file driver --------------------------------------------------------

def process_one(client: OpenAI, src: Path, outdir: Path, do_format: bool) -> Path | None:
    if not src.exists():
        err(f"File not found: {src}")
        return None
    if src.is_dir():
        err(f"That's a folder, not a file: {src}")
        return None
    if src.suffix.lower() not in MEDIA_EXTS:
        warn(f"'{src.name}' doesn't look like an audio/video file — trying anyway.")

    print()
    step(f"Transcribing: {src.name}")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            audio_path = workdir / "audio.mp3"
            info("Extracting & compressing audio…")
            extract_audio(src, audio_path)
            transcript = transcribe_any(client, audio_path, workdir)
    except RuntimeError as e:
        err(str(e))
        return None
    except Exception as e:  # noqa: BLE001 - surface anything cleanly per file
        err(f"Could not transcribe {src.name}: {e}")
        return None

    if not transcript.strip():
        warn(f"Whisper returned an empty transcript for {src.name}. Skipping.")
        return None

    stem = safe_filename(src.stem)
    out_path = unique_path(outdir, stem)
    out_path.write_text(transcript, encoding="utf-8")
    ok(f"Saved transcript → {out_path}")

    if do_format:
        try:
            cleaned = format_transcript(client, transcript)
            if cleaned.strip():
                fmt_path = unique_path(outdir, f"{stem} (formatted)")
                fmt_path.write_text(cleaned, encoding="utf-8")
                ok(f"Saved cleaned copy → {fmt_path}")
        except Exception as e:  # noqa: BLE001
            warn(f"Could not GPT-format {src.name} (raw transcript is still saved): {e}")

    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Drag-and-drop audio/video → text transcriber (OpenAI Whisper).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("files", nargs="*", help="Audio/video file(s) to transcribe.")
    parser.add_argument(
        "--format", "-f", action="store_true",
        help="Also produce a GPT-cleaned copy (punctuation, paragraphs, fixes).",
    )
    parser.add_argument(
        "--outdir", "-o", default=None,
        help="Where to save transcripts (default: your Desktop).",
    )
    args = parser.parse_args()

    print(_c("\n  🎙  Qargo Transcribe (CLI)\n", "1"))

    ensure_ffmpeg()

    outdir = Path(os.path.expanduser(args.outdir)) if args.outdir else default_output_dir()
    outdir.mkdir(parents=True, exist_ok=True)

    paths = [Path(os.path.expanduser(f)) for f in args.files]
    if not paths:
        paths = collect_paths_interactively()
    if not paths:
        warn("No files given. Nothing to do.")
        return 0

    do_format = args.format
    if not do_format and sys.stdin.isatty():
        try:
            ans = input("\n  Also create a cleaned-up version with GPT? [y/N]: ").strip().lower()
            do_format = ans in {"y", "yes"}
        except (EOFError, KeyboardInterrupt):
            print()

    client = OpenAI(api_key=get_api_key())

    info(f"Saving results to: {outdir}")
    succeeded = 0
    for src in paths:
        if process_one(client, src, outdir, do_format) is not None:
            succeeded += 1

    print()
    if succeeded == len(paths):
        ok(f"All done — {succeeded}/{len(paths)} file(s) transcribed.")
    else:
        warn(f"Finished — {succeeded}/{len(paths)} file(s) transcribed, "
             f"{len(paths) - succeeded} skipped.")
    return 0 if succeeded else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        sys.exit(130)
