# Video → Text

A tiny Streamlit app that turns a video file, audio file, or YouTube URL into a
plain-text transcript using OpenAI's Whisper API.

## Use it

1. Upload an audio/video file **or** paste a URL.
2. Click **Transcribe**.
3. Read the transcript in the app, or download it as `.txt`.

## Run locally

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
streamlit run streamlit_app.py
```

You also need `ffmpeg` installed on your machine (`brew install ffmpeg` on macOS,
`sudo apt install ffmpeg` on Ubuntu).

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing
   at this repo and the `streamlit_app.py` file.
3. Under **Settings → Secrets**, add:

   ```toml
   OPENAI_API_KEY = "sk-..."
   ```

4. `packages.txt` tells Streamlit Cloud to install `ffmpeg` on the host.

## How it works

- Any input (video or YouTube) is normalized with `ffmpeg` to a mono 16 kHz MP3.
- If the resulting audio is under Whisper's 25 MB upload limit, it's sent as-is.
- If it's larger, it's split into 10-minute chunks and transcribed sequentially,
  then the parts are concatenated.
- Language is auto-detected by Whisper.
