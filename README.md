# 🎙️ Video → Text

A tiny Streamlit app that turns a video file, audio file, or YouTube URL into
a plain-text transcript using OpenAI's Whisper API.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

## Use it

1. Upload an audio/video file **or** paste a URL.
2. Click **Transcribe**.
3. Read the transcript in the app, or download it as `.txt`.

## Run it on your own machine

1. Install the requirements
   ```bash
   pip install -r requirements.txt
   ```

2. Make sure `ffmpeg` is installed
   ```bash
   # macOS
   brew install ffmpeg
   # Ubuntu/Debian
   sudo apt install ffmpeg
   ```

3. Set your OpenAI key and run the app
   ```bash
   export OPENAI_API_KEY=sk-...
   streamlit run streamlit_app.py
   ```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing
   at this repo and the `streamlit_app.py` file.
3. Under **Settings → Secrets**, add:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```

`packages.txt` tells Streamlit Cloud to install `ffmpeg` on the host.

## How it works

- Any input (video or YouTube) is normalized with `ffmpeg` to a mono 16 kHz MP3.
- If the resulting audio is under Whisper's 25 MB upload limit, it's sent as-is.
- If it's larger, it's split into 10-minute chunks and transcribed sequentially,
  then the parts are concatenated.
- Language is auto-detected by Whisper.
