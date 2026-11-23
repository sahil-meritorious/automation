# AI YouTube Shorts Generator

Paste a YouTube URL (or a local video file) that you own or are authorised to use. The app transcribes it with Groq Whisper and uses an LLM to find the most entertaining moments. It then cuts those moments into captioned vertical 1080×1920 Shorts, checks every file, and exports them with titles, descriptions, hashtags and a report.

> **Rights:** only process videos you own or have permission to edit and republish. The app uses yt-dlp's normal download path. It does not bypass logins, access controls or DRM. Background music must be audio you have the rights to use.

---

## 1. Architecture and workflow

```
URL / file ─► Downloader ─► ffprobe metadata ─► Audio extract (mono 16 kHz WAV)
          ─► Chunker (10-min FLAC chunks, 5 s overlap, < 20 MB each)
          ─► Groq Whisper (verbose_json, word + segment timestamps, rate-limited, cached per chunk)
          ─► Transcript merger (offsets restored, overlap de-duplicated, hallucinations filtered)
          ─► Highlight detector (LLM, 10-min sections with 60 s overlap, strict JSON, cached per section)
          ─► Ranker (snap to word/sentence boundaries, enforce durations, score, de-duplicate,
                     optional global LLM re-rank, quality threshold, no filler)
          ─► Metadata generator (titles, descriptions, hashtags, filenames)
          ─► For each Short (in parallel): clip extract ─► 9:16 layout ─► punch-in/zoom/fades
                     ─► ASS animated captions ─► loudness normalisation / music ─► H.264 + AAC
                     ─► Quality check (codecs, 1080×1920, duration, A/V drift, decode test, caption sync)
          ─► output/<video>/  01_title.mp4 + .json + .jpg, metadata.json/.csv, report.json/.md
```

**Checkpointing:** every stage writes its result to disk, and the SQLite DB (`database/shorts.db`) tracks job status. If anything fails or you cancel, re-running the same URL (or pressing **Resume**) skips the download, the transcribed chunks, the analysed sections and any Shorts that were already rendered. Expensive API work is never repeated, and changing an LLM or render setting only redoes the stages that depend on it.

### Project layout

```
app.py                    Streamlit UI
main.py                   CLI: process / resume / jobs / self-test
config.py                 All settings (.env + UI/CLI overrides), validated by Pydantic
core/
  pipeline.py             Orchestrator, checkpoints, progress, parallel rendering, QC-gated export
  job_runner.py           Background jobs for the UI (cancel / resume)
  llm_client.py           OpenAI-compatible client: Gemini, OpenAI, Groq, OpenRouter, Ollama, custom
  rate_limiter.py         Sliding-window limiter (RPM + audio-seconds/hour/day), shared across jobs
  models.py               Pydantic models (transcript, candidates, metadata)
  utils.py, logging_setup.py
modules/
  downloader.py           yt-dlp + local files, URL validation, caching, retries, disk checks
  video_metadata.py       ffprobe wrapper
  audio_extractor.py      FFmpeg audio extraction
  audio_chunker.py        Overlapping chunks under the upload limit (FLAC → MP3 fallback)
  transcription.py        Groq Whisper REST client: retries, 429 handling, per-chunk cache
  transcript_merger.py    Timeline merge + overlap de-duplication
  highlight_detector.py   Windowed LLM analysis → candidates
  highlight_ranker.py     Boundary refinement, scoring, de-dup, global re-rank, selection
  clip_extractor.py       Frame-accurate segment extraction
  vertical_editor.py      9:16 layouts, effects, audio, final encode
  caption_generator.py    ASS captions: dynamic / minimal / karaoke (+ SRT track for YouTube)
  metadata_generator.py   Titles, descriptions, hashtags, tags, JSON/CSV/Markdown reports
  quality_checker.py      Export validation
  trend_research.py       YouTube Data API trending keywords (API key, cached daily)
  youtube_uploader.py     OAuth, resumable upload, scheduling, captions, quota handling
database/database.py      SQLite: videos, jobs, shorts (+ upload status), YouTube quota
prompts/                  highlight_detection.txt, highlight_rerank.txt, metadata_generation.txt
tests/                    Unit tests + offline end-to-end test (APIs mocked)
input/ downloads/ transcripts/ highlights/ clips/ output/ logs/
```

---

## 2a. Installation on Linux (Ubuntu / Debian)

```bash
# System packages: FFmpeg, Python venv support, fonts
sudo apt update
sudo apt install -y ffmpeg python3-venv python3-pip fonts-liberation
ffmpeg -version && python3 --version          # Python 3.10+ is fine

# Deno: the JavaScript runtime yt-dlp needs for YouTube
curl -fsSL https://deno.land/install.sh | sh
echo 'export PATH="$HOME/.deno/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
deno --version

# The project
cd ~/youtube-automation
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
nano .env        # add GROQ_API_KEY and your LLM key (section 3)
```

Run all later commands with the venv active (`source .venv/bin/activate`). Then use `python` exactly as in the Windows examples below, with `/` paths, for example `python main.py process ~/Videos/session1.mp4 --confirm-rights`.
If `Arial Black` isn't installed (`fc-match "Arial Black"` shows another font), either set `CAPTION_FONT=Liberation Sans` or copy a `.ttf` into `assets/fonts/`.

## 2b. Installation on Windows

### 2.1 Python 3.11 or newer
1. Download Python 3.11+ from https://www.python.org/downloads/windows/ (or run `winget install -e --id Python.Python.3.12`).
2. During setup, tick **"Add python.exe to PATH"**.
3. Check it in a new terminal: `python --version`

### 2.2 FFmpeg
Easiest (Windows 10/11):
```powershell
winget install -e --id Gyan.FFmpeg
```
Close and reopen the terminal, then check: `ffmpeg -version` and `ffprobe -version`.

Manual alternative: download the "release full" build from https://www.gyan.dev/ffmpeg/builds/ and extract it to `C:\ffmpeg`. Add `C:\ffmpeg\bin` to PATH, or set `FFMPEG_PATH=C:\ffmpeg\bin\ffmpeg.exe` and `FFPROBE_PATH=C:\ffmpeg\bin\ffprobe.exe` in `.env`.

### 2.3 Deno (JavaScript runtime that yt-dlp now needs for YouTube)
```powershell
winget install -e --id DenoLand.Deno
```

### 2.4 The project
```powershell
cd C:\path\to\youtube-automation
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
notepad .env
```
(If PowerShell refuses to activate the venv, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.)

---

## 3. API keys

### Groq (transcription, required, has a free tier)
1. Sign in at https://console.groq.com.
2. Open **API Keys → Create API Key**, then copy it.
3. In `.env`, set `GROQ_API_KEY=gsk_...`

The free tier has limits of roughly 20 requests/min, 7,200 audio-seconds/hour, 28,800 audio-seconds/day and 25 MB per file. The defaults stay under these limits, so a 2-hour video takes about one hour of quota. If Groq returns 429, the app waits for the time Groq asks for and then continues. On a paid tier, raise `GROQ_*` limits and `TRANSCRIPTION_WORKERS`.

### LLM for highlight detection (pick one)
| Provider | `.env` | Get a key |
|---|---|---|
| Gemini (default, has a free tier) | `LLM_PROVIDER=gemini`, `GEMINI_API_KEY=...` | https://aistudio.google.com/apikey |
| OpenAI | `LLM_PROVIDER=openai`, `OPENAI_API_KEY=...` | https://platform.openai.com/api-keys |
| Groq-hosted open models | `LLM_PROVIDER=groq` (reuses `GROQ_API_KEY`) | same as above |
| OpenRouter | `LLM_PROVIDER=openrouter`, `OPENROUTER_API_KEY=...` | https://openrouter.ai/keys |
| Local Ollama | `LLM_PROVIDER=ollama`, `LLM_MODEL=qwen2.5:14b` | no key |
| Any OpenAI-compatible server | `LLM_PROVIDER=custom`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | — |

Leave `LLM_MODEL` blank to use the provider default (see `PROVIDERS` in `core/llm_client.py`). Model names change often, so set `LLM_MODEL` explicitly if a default is retired. You can also enter keys in the app sidebar and click **Save to .env**.

---

## 4. Launching

**Web UI**
```powershell
.venv\Scripts\activate
streamlit run app.py
```
Your browser opens at http://localhost:8501. Paste the URL, tick the rights checkbox and click **Start processing**. You can start another URL at any time without restarting.

**CLI and batch**
```powershell
python main.py process "https://www.youtube.com/watch?v=VIDEO_ID" --confirm-rights
python main.py process "D:\Recordings\session1.mp4" --confirm-rights --shorts 5 --caption-style karaoke
python main.py process -f urls.txt --confirm-rights            # one URL/path per line, '#' comments allowed
python main.py process -f urls.txt --confirm-rights --parallel 2
python main.py jobs                                            # list jobs and their status
python main.py resume <JOB_ID>                                 # continue an interrupted/failed job
python main.py process --help                                  # all options
```

---

## 4b. AI direction and topic mode

**The AI decides everything by default.** For every Short it chooses:
- the cut: setup, hook and payoff
- the on-screen hook line and the caption style
- the layout: full frame, centre crop or face tracking
- punch-in and slow zoom
- the title, description, hashtags and tags

Every prompt optimises for virality with your `TARGET_AUDIENCE`, which defaults to GTA 6 / gaming fans. The **AI decisions** tab shows each choice and why the AI made it.

Any sidebar setting left on **Auto** stays with the AI. Change one and your choice wins; it is marked "you". **Save as default** writes your choices to `.env`.

**Topic mode.** Choose **Topic** in the source switch and type a topic such as `GTA 5 vs GTA 6 comparison` or `GTA 6 easter eggs`. The app then:
1. Searches YouTube with your API key.
2. Keeps videos 4–120 minutes long that you're allowed to reuse.
3. Lets the AI pick the one with the most viral potential.
4. Creates the Shorts.

Already-processed videos are skipped. From the CLI:
```bash
python main.py process --topic "GTA 6 easter eggs" --videos 2 --confirm-rights
```
Where it may search (`TOPIC_SOURCE`):
- `creative_commons` (default): CC BY videos. A credit line is added to every description, and likely re-uploads of other creators are skipped, because their CC label is often invalid.
- `my_channel`: your connected channel.
- `allowed_channels`: the channel IDs in `ALLOWED_CHANNEL_IDS` that you have permission from.

Still check the picked video before publishing publicly. A CC label is the uploader's claim, not proof that they own the footage.

## 4c. Clipping campaigns

Choose **Campaign** in the source switch and paste the brand's brief as-is. You can also upload the brand logo and add extra video links. Click **Analyze requirements** to see what the AI understood, or **Analyze & create clips** to start right away.

The AI reads the brief into a checklist, and the pipeline follows it:

| In the brief | What happens |
|---|---|
| Video links | One project per video |
| Exact timings ("use 2:10–2:45") | Cut exactly there. The AI only finds the hook inside the segment and chooses the title, on-screen hook line and edit |
| "Add N more" | N extra hook-based clips, never overlapping the given segments |
| No timings | Hook-based detection, guided by the brief's focus, dos and don'ts |
| Clip count and length | Sets the number of clips and the min/max length |
| Required hashtags, @mentions, caption text, title text | Always added to the metadata, after the AI writes it |
| Logo | Overlaid on every clip in the chosen corner, clear of the Shorts UI |
| On-screen text ("link in bio") | Stays visible for the whole clip |
| Subtitles required / no music | Captions forced on / background music disabled |

Each clip's checklist appears in the **AI decisions** tab. Each campaign's files go to `output/<campaign>/`, and the brief and logo are saved in `campaigns/<campaign>/`.

From the CLI:
```bash
python main.py process --brief campaign.txt --logo logo.png --confirm-rights
```

## 5. Test procedure (do this before processing long videos)

1. **Editing chain, no API keys needed:**
   ```powershell
   python main.py self-test
   ```
   This renders three 20-second test Shorts (one per caption style) into `output\_selftest\`. Open them and check the captions, the title and the punch-in at 3 s.
2. **API keys:**
   ```powershell
   python main.py self-test --with-apis
   ```
3. **Automated tests:**
   ```powershell
   python -m unittest discover -s tests -v
   ```
   This includes a full offline pipeline run with Groq and the LLM mocked.
4. **A real short video:** use a 3–10 minute video you own, with `--shorts 3`:
   ```powershell
   python main.py process "<url>" --confirm-rights --shorts 3
   ```
   Check `output\<id>_<title>\report.md`, the transcript in `transcripts\<id>\transcript.txt` and the Shorts.
5. Only then process multi-hour videos. To confirm caching works, run the same URL again: it should finish in seconds.

---

## 6. Tuning tips

- **`CONTENT_CONTEXT`** strongly affects judgement. Describe the channel, for example "GTA 6 story-mode playthrough with funny commentary by two friends".
- **`MIN_SCORE`** is the quality bar. If too few Shorts come out, lower it to 55. If weak clips get through, raise it to 70.
- **Layouts:** `blur_fit` keeps the whole 16:9 frame, including the HUD and minimap, which suits gameplay. `FOREGROUND_ZOOM=1.3` makes the action bigger but crops the sides. Use `offset_crop` with `CROP_X_POSITION` for a fixed region such as a facecam on the right. `face_track` (needs `opencv-python-headless`) finds the main face and centres a stable crop on it, and falls back to `blur_fit` when there is no face.
- **Captions** are placed below the gameplay band in `blur_fit` (outside the action and out of the Shorts UI area). Put `.ttf` fonts in `assets/fonts/` and set `CAPTION_FONT` to the font's family name.
- **Speed:** `VIDEO_PRESET=veryfast` renders about 3× faster with slightly larger files.

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `Executable not found: 'ffmpeg'` | Install FFmpeg (2.2), reopen the terminal, or set `FFMPEG_PATH`/`FFPROBE_PATH`. |
| YouTube download fails: "Sign in to confirm you're not a bot", "n challenge", HTTP 403, missing formats | Run `pip install -U "yt-dlp[default]"` and install Deno (2.3). If it still fails, download your own video another way (for example YouTube Studio → Download) and pass the local file path. |
| `Groq rejected the API key` | Check `GROQ_API_KEY` and remove spaces or quotes. |
| Long "rate limit: waiting …" messages | You hit the Groq free-tier limit. The job continues by itself. Progress is cached, so you can also stop it and resume later. |
| `Chunk … exceeds Groq's upload limit` | Lower `AUDIO_CHUNK_SECONDS` (for example 300). |
| `LLM authentication failed` / `model not found` | Wrong key, or a retired model. Set `LLM_MODEL` to a current model. |
| `invalid JSON from LLM` warnings | Handled automatically (the reply is repaired, then the request retried). If a section still fails, it is skipped and retried on the next run. A stronger model also helps. |
| "No moments met the quality bar" | Lower `MIN_SCORE`, improve `CONTENT_CONTEXT`, or check that the transcript has speech. |
| Captions missing / wrong font / `exported WITHOUT captions` | The font isn't installed. Put the `.ttf` in `assets/fonts/` or use `CAPTION_FONT=Arial`. |
| `Not enough disk space` | Free some space or lower `MIN_FREE_DISK_GB`. Delete old `downloads\` folders you no longer need. |
| Streamlit shows an old job as interrupted | The app was closed mid-job. Select it and click **Resume job**. |
| Need details | `logs\app.log` (everything) and `logs\jobs\<job_id>.log` (one job). |

---

## 8. YouTube upload and trending metadata

Two different Google credentials do two different jobs:

| Credential | Used for | Required? |
|---|---|---|
| **API key** (`YOUTUBE_API_KEY`) | Read-only trend research: what recent Shorts on your topic rank for | Optional |
| **OAuth client ID + secret** | Uploading to *your* channel (an API key **cannot** upload) | Required for upload |

### 8.1 One-time Google Cloud setup
1. Go to [console.cloud.google.com](https://console.cloud.google.com), pick your project, then open **APIs & Services → Library**. Enable **YouTube Data API v3**.
2. Open **OAuth consent screen**. Choose type *External*, fill in the app name and your email, and add your own Google account under **Test users**.
3. Open **Credentials → Create credentials → OAuth client ID**. Pick application type **Desktop app**, then copy the **Client ID** and **Client secret**.
   - Under **Authorized redirect URIs**, add `http://localhost:8501/` (the Streamlit app; the trailing slash matters). For `python main.py youtube-auth`, also add `http://localhost:8765/`.
   - If you see **access_denied**, add your channel's Google account under **OAuth consent screen → Audience → Test users**.
4. Put the credentials in `.env`, or enter them in the sidebar under **📺 YouTube** and press **Save YouTube settings**:
   ```
   YOUTUBE_API_KEY=AIza...
   YOUTUBE_CLIENT_ID=1234-abc.apps.googleusercontent.com
   YOUTUBE_CLIENT_SECRET=GOCSPX-...
   ```
5. Connect your channel. This step is needed once:
   ```bash
   python main.py youtube-auth      # opens the browser; sign in with the channel's Google account
   python main.py youtube-status    # checks the connection, API key and quota
   ```
   You can also press **🔗 Connect YouTube** in the job's **📤 YouTube** tab. The refresh token is stored in `database/youtube_token.json`, which is git-ignored and readable only by your user.

**Two Google restrictions to know:**
- **Testing-mode tokens expire after 7 days.** While the consent screen is in *Testing*, you have to reconnect every week. Press **Publish app** on the consent screen to avoid that. For personal use, Google will show an "unverified app" warning; click *Advanced → Continue*.
- **Unverified projects can only upload privately.** Uploads through an unaudited API project created after July 2020 are locked to *private*. To publish as public or unlisted from the API, request the [YouTube API audit](https://support.google.com/youtube/contact/yt_api_form). Until it is approved, upload as private and switch the visibility in YouTube Studio.

### 8.2 Uploading
- **UI:** open a finished job and go to the **📤 YouTube** tab.
  1. Review and edit the title, description, hashtags and tags, then press **Save**.
  2. Pick the Shorts, the privacy setting and an optional schedule, then press **Upload to YouTube**.
- **CLI:**
  ```bash
  python main.py upload JOB_ID                                 # default privacy (YOUTUBE_PRIVACY)
  python main.py upload JOB_ID --privacy public --interval-hours 3   # one Short goes live every 3 h
  python main.py upload JOB_ID --ranks 1,2 --privacy unlisted
  python main.py process URL --confirm-rights --upload --privacy public --interval-hours 4
  ```
- **Fully automatic:** set `AUTO_UPLOAD=true`, or tick *Upload automatically* in the sidebar.

Each upload sets:
- The title, plus a description made of your text, `YOUTUBE_DESCRIPTION_FOOTER` and 3–5 hashtags. The first 3 hashtags appear above the title.
- Up to roughly 450 characters of search **tags**, the category (default 20, Gaming), the language and the made-for-kids flag.
- An **SRT subtitle track** built from the Whisper transcript, so the spoken words become searchable text.
- Optionally a custom thumbnail (`YOUTUBE_SET_THUMBNAIL=true`, which needs a phone-verified channel).

Uploads are resumable and retried. A Short that is already uploaded is skipped, unless you pass `--force`. Results go to the database and to `output/<video>/uploads.json`.

**Quota:** the default allowance is 10,000 units per day, and it resets at midnight Pacific time. Google lowered the upload cost to about 100 units, down from 1,600; a caption track costs 400 units and a trend search about 100. If the quota runs out, the batch stops cleanly and the remaining Shorts are marked *pending*. Run the same upload command the next day. The quota counter shown in the app is only an estimate; the API is authoritative.

### 8.3 How the metadata is optimised for reach
1. **Trend research.** With an API key, the app runs `search.list` for Shorts on your topic from the last 14 days, sorted by views, plus the category's *mostPopular* chart.
   - The topic comes from `TREND_QUERY`, or is derived from the source title (for example "GTA 6").
   - Tags, hashtags and title phrases are weighted by views and frequency, then cached for the day.
   - Try it: `python main.py trends "GTA 6" --region IN`
2. **The LLM writes the metadata** using the transcript of each clip, the source video's tags and the trending keywords:
   - A title of 40–70 characters with the searchable keyword first.
   - A keyword-rich description that ends with a question, to drive comments.
   - 3–5 hashtags and 8–15 long-tail tags.
3. **Relevance is enforced.** The prompt only allows trending terms that genuinely describe the clip. Irrelevant tags or misleading titles break YouTube's spam policy and get Shorts suppressed rather than promoted.

Metadata drives discovery, but reach on Shorts mostly comes from retention: the hook in the first second, plus the percentage of viewers who watch to the end or rewatch. That is what the highlight scoring optimises.

To get trend-optimised metadata for an older job, process the same URL again. The cached download, transcript and analysis are reused; only the metadata and the Shorts with changed titles are regenerated.

---

## 9. Honest limitations

- Highlight detection uses the transcript only. The AI infers visual events from what people say and how they react, so a silent visual gag with no verbal reaction can be missed.
- `face_track` produces one stable crop per clip (the median face position), not a moving camera. This avoids jitter, but it will not follow someone who walks across the frame.
- Word timestamps come from Whisper. Captions can drift by a few hundred milliseconds in music-heavy passages. The QC step warns when caption timing doesn't line up with detected speech.
