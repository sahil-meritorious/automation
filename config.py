"""Central configuration.

All settings can be supplied through environment variables or the ``.env`` file
(variable names are the upper-case field names, e.g. ``GROQ_API_KEY``).
Runtime overrides (from the Streamlit UI or the CLI) are passed as keyword
arguments to :func:`get_settings`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent

CAPTION_STYLES = ("dynamic", "minimal", "karaoke")
LAYOUTS = ("blur_fit", "center_crop", "offset_crop", "face_track")
RESOLUTIONS = {"1080x1920": (1080, 1920), "720x1280": (720, 1280)}
TRANSCRIPTION_MODELS = ("whisper-large-v3-turbo", "whisper-large-v3")
LLM_PROVIDERS = ("gemini", "openai", "groq", "openrouter", "ollama", "custom")
PRIVACY_OPTIONS = ("private", "unlisted", "public")
AUTO = "auto"  # "let the AI decide" value for creative settings
TOGGLE_MODES = ("auto", "on", "off")
TOPIC_SOURCES = ("any_public", "allowed_channels", "creative_commons", "my_channel")
SECRET_MARKERS = ("key", "secret", "token", "client_id")
# Account-level settings that always come from the current .env / sidebar, never from a saved job
ACCOUNT_FIELDS = ("llm_provider", "llm_model", "llm_base_url", "transcription_model", "ffmpeg_path", "ffprobe_path",
                  "data_dir")


def resume_overrides(saved: dict, current: Optional[dict] = None) -> dict:
    """Saved job settings minus account fields, plus current account fields/secrets (if given)."""
    out = {k: v for k, v in saved.items() if k not in ACCOUNT_FIELDS}
    for k, v in (current or {}).items():
        if (k in ACCOUNT_FIELDS or any(m in k for m in SECRET_MARKERS)) and v not in (None, ""):
            out[k] = v
    out.pop("data_dir", None)
    return out


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
        validate_assignment=True,
    )

    # ------------------------------------------------------------------ keys
    groq_api_key: str = ""
    # extra Groq keys: when one key is rate-limited the app switches to the next immediately
    groq_api_key_1: str = ""
    groq_api_key_2: str = ""
    groq_api_key_3: str = ""
    groq_api_key_4: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""
    openrouter_api_key: str = ""
    llm_api_key: str = ""  # used by the "custom" provider, or to override any provider's key

    # --------------------------------------------------------- transcription
    transcription_model: str = "whisper-large-v3-turbo"
    transcription_language: str = ""  # ISO-639-1 code such as "en"; empty = auto-detect
    transcription_prompt: str = ""  # optional spelling hints, e.g. "GTA, Los Santos, Vice City"
    groq_max_upload_mb: float = 25.0  # Groq free tier limit; dev tier is 100 MB
    upload_safety_margin: float = 0.8  # chunks must stay below limit * margin
    audio_chunk_seconds: int = 600  # 10-minute chunks
    audio_chunk_overlap: float = 5.0
    audio_chunk_format: str = "flac"  # flac (lossless, ~50% of wav) or mp3
    groq_requests_per_minute: int = 18  # free tier: 20 RPM
    groq_audio_seconds_per_hour: int = 7000  # free tier: 7200 audio seconds / hour
    groq_audio_seconds_per_day: int = 28000  # free tier: 28800 audio seconds / day
    transcription_workers: int = 1  # parallel chunk uploads (keep 1 on the free tier)
    transcription_max_retries: int = 8

    # ------------------------------------------------------------------- LLM
    llm_provider: str = "groq"
    llm_model: str = ""  # empty = provider default (see core/llm_client.py)
    llm_base_url: str = ""  # required for "custom", optional override otherwise
    llm_temperature: Optional[float] = 0.4
    llm_max_concurrency: int = 2
    llm_min_interval: float = 1.0  # seconds between LLM requests
    llm_rerank: bool = True  # second, global LLM pass that compares all candidates
    llm_tokens_per_minute: int = 0  # pace requests by tokens; 0 = auto (Groq free tier: 7000), -1 = off
    llm_max_request_tokens: int = 0  # split prompts above this size; 0 = auto (Groq: 5500), -1 = off

    # ----------------------------------------------------- highlight detection
    content_context: str = "GTA gameplay with live commentary"
    target_audience: str = (
        "Gaming fans on YouTube Shorts, mostly 16-30, who follow GTA 6 / GTA 5 / Rockstar news, "
        "easter eggs, funny moments and streamer reactions; they swipe away in under 2 seconds")
    analysis_window_seconds: int = 600
    analysis_overlap_seconds: int = 60
    candidates_per_window: int = 6
    max_setup_seconds: float = 12.0  # max context kept before the hook when trimming

    # ------------------------------------------------------------ selection
    num_shorts: int = 10
    min_clip_duration: float = 15.0
    max_clip_duration: float = 60.0
    preferred_min_duration: float = 25.0
    preferred_max_duration: float = 45.0
    duration_tolerance: float = 0.15  # may exceed max by this fraction to keep a payoff
    min_score: float = 60.0

    # ---------------------------------------------------------------- video
    output_width: int = 1080
    output_height: int = 1920
    fps: int = 30
    video_crf: int = 20
    video_preset: str = "medium"
    audio_bitrate: str = "192k"
    layout: str = "auto"  # auto = AI picks per clip; or blur_fit | center_crop | offset_crop | face_track
    crop_x_position: float = 0.5  # 0 = left edge, 1 = right edge (crop layouts)
    foreground_zoom: float = 1.0  # blur_fit: >1 enlarges the gameplay (crops sides)
    max_download_height: int = 1080

    # ------------------------------------------------------------- captions
    captions_enabled: bool = True
    caption_style: str = "auto"  # auto = AI picks per clip; or dynamic | minimal | karaoke
    caption_font: str = "Arial Black"
    caption_font_size: int = 0  # 0 = style default
    caption_position: float = 0.0  # fraction of height for caption centre; 0 = automatic
    title_overlay: bool = True
    use_llm_caption_style: bool = False  # legacy; caption_style=auto replaces it
    overlay_text: str = "auto"  # on-screen hook text: auto (AI hook line) | title | off

    # -------------------------------------------------------------- effects
    zoom_mode: str = "auto"  # auto | on | off  (slow zoom over the whole clip)
    punch_in_mode: str = "auto"  # auto | on | off  (quick zoom at the hook)
    zoom_effect: bool = False  # resolved per clip from zoom_mode
    zoom_max: float = 1.06
    punch_in: bool = True  # resolved per clip from punch_in_mode
    punch_in_zoom: float = 1.12
    fade_transitions: bool = True
    normalize_audio: bool = True
    target_lufs: float = -14.0
    background_music: str = ""  # file or folder of music you have rights to use
    music_volume: float = 0.08

    # -------------------------------------------------------------- YouTube
    youtube_api_key: str = ""  # Data API v3 key — used for trending-keyword research (read-only)
    youtube_client_id: str = ""  # OAuth client (Desktop app) — needed to upload
    youtube_client_secret: str = ""
    youtube_client_secrets_file: str = ""  # alternative: path to the downloaded client_secret_*.json
    youtube_oauth_port: int = 8765  # CLI sign-in only: local redirect port
    youtube_redirect_uri: str = "http://localhost:8501/"  # Streamlit sign-in: must match the OAuth client exactly
    youtube_privacy: str = "private"  # private | unlisted | public
    youtube_category_id: str = "20"  # 20 = Gaming, 22 = People & Blogs, 24 = Entertainment
    youtube_made_for_kids: bool = False
    youtube_region: str = "US"  # region used for trending research
    youtube_language: str = "en"  # default language of titles/descriptions and audio
    youtube_upload_captions: bool = True  # also upload an SRT subtitle track (searchable text)
    youtube_set_thumbnail: bool = False  # needs a phone-verified channel
    youtube_schedule_interval_hours: float = 0.0  # >0 = publish one Short every N hours (scheduled)
    youtube_description_footer: str = ""  # appended to every description (links, credits)
    youtube_daily_quota: int = 10000  # informational counter only; the API is authoritative
    youtube_upload_cost: int = 100  # quota units per videos.insert (Google lowered it from 1600)
    youtube_caption_cost: int = 400
    auto_upload: bool = False  # upload every exported Short when a job finishes
    trend_research: bool = True  # use YOUTUBE_API_KEY to feed trending keywords into metadata
    trend_query: str = ""  # search phrase for trend research; blank = derived from the video title
    trend_days: int = 14  # look at videos published in the last N days

    # ------------------------------------------------------- topic discovery
    topic_source: str = "creative_commons"  # any_public | allowed_channels | creative_commons | my_channel
    allowed_channel_ids: str = ""  # comma-separated channel IDs (UC...) or @handles you have permission to clip
    private_only: bool = False  # set automatically for test jobs from any_public: uploads are forced private
    topic_min_minutes: float = 4.0
    topic_max_minutes: float = 120.0

    # ------------------------------------------------------------- campaigns
    campaign_json: str = ""  # parsed campaign brief (set per job by the Campaign mode)
    campaign_slug: str = ""
    logo_path: str = ""  # brand logo/watermark image (png/jpg/webp) overlaid on every clip
    logo_position: str = "top_right"  # top_right | top_left | top_center | bottom_right | bottom_left
    logo_scale: float = 0.20  # logo width as a fraction of the video width
    logo_opacity: float = 1.0
    on_screen_text: str = ""  # text that must stay visible for the whole clip (e.g. "Link in bio")

    # ---------------------------------------------------------- reliability
    download_retries: int = 3
    render_workers: int = 2
    keep_intermediate: bool = False
    min_free_disk_gb: float = 2.0
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"

    data_dir: Path = Field(default=BASE_DIR)

    # ------------------------------------------------------------ validators
    @field_validator("caption_style")
    @classmethod
    def _caption_style(cls, v: str) -> str:
        v = (v or AUTO).strip().lower()
        if v not in CAPTION_STYLES + (AUTO,):
            raise ValueError(f"caption_style must be one of {CAPTION_STYLES}")
        return v

    @field_validator("layout")
    @classmethod
    def _layout(cls, v: str) -> str:
        v = (v or AUTO).strip().lower()
        if v not in LAYOUTS + (AUTO,):
            raise ValueError(f"layout must be one of {LAYOUTS}")
        return v

    @field_validator("llm_provider")
    @classmethod
    def _provider(cls, v: str) -> str:
        v = (v or "groq").strip().lower()
        if v not in LLM_PROVIDERS:
            raise ValueError(f"llm_provider must be one of {LLM_PROVIDERS}")
        return v

    @field_validator("zoom_mode", "punch_in_mode")
    @classmethod
    def _toggle(cls, v: str) -> str:
        v = str(v).strip().lower()
        v = {"true": "on", "1": "on", "yes": "on", "false": "off", "0": "off", "no": "off"}.get(v, v)
        if v not in TOGGLE_MODES:
            raise ValueError(f"must be one of {TOGGLE_MODES}")
        return v

    @field_validator("overlay_text")
    @classmethod
    def _overlay(cls, v: str) -> str:
        v = (v or AUTO).strip().lower()
        if v not in (AUTO, "title", "off"):
            raise ValueError("overlay_text must be auto, title or off")
        return v

    @field_validator("topic_source")
    @classmethod
    def _topic_source(cls, v: str) -> str:
        v = (v or "creative_commons").strip().lower()
        if v not in TOPIC_SOURCES:
            raise ValueError(f"topic_source must be one of {TOPIC_SOURCES}")
        return v

    @field_validator("youtube_privacy")
    @classmethod
    def _privacy(cls, v: str) -> str:
        v = (v or "private").strip().lower()
        if v not in PRIVACY_OPTIONS:
            raise ValueError(f"youtube_privacy must be one of {PRIVACY_OPTIONS}")
        return v

    @field_validator("crop_x_position")
    @classmethod
    def _crop_pos(cls, v: float) -> float:
        return min(1.0, max(0.0, float(v)))

    @field_validator("output_width", "output_height")
    @classmethod
    def _even(cls, v: int) -> int:
        v = int(v)
        return v - (v % 2)

    # ----------------------------------------------------------------- paths
    @property
    def input_dir(self) -> Path:
        return self.data_dir / "input"

    @property
    def downloads_dir(self) -> Path:
        return self.data_dir / "downloads"

    @property
    def transcripts_dir(self) -> Path:
        return self.data_dir / "transcripts"

    @property
    def highlights_dir(self) -> Path:
        return self.data_dir / "highlights"

    @property
    def clips_dir(self) -> Path:
        return self.data_dir / "clips"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "output"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "database" / "shorts.db"

    @property
    def prompts_dir(self) -> Path:
        return BASE_DIR / "prompts"

    @property
    def fonts_dir(self) -> Path:
        return BASE_DIR / "assets" / "fonts"

    @property
    def youtube_token_path(self) -> Path:
        return self.data_dir / "database" / "youtube_token.json"

    @property
    def groq_keys(self) -> list:
        """All configured Groq keys, primary first, duplicates removed."""
        keys = [self.groq_api_key, self.groq_api_key_1, self.groq_api_key_2, self.groq_api_key_3,
                self.groq_api_key_4]
        return list(dict.fromkeys(k.strip() for k in keys if k and k.strip()))

    @property
    def effective_llm_tpm(self) -> int:
        if self.llm_tokens_per_minute:
            return max(0, self.llm_tokens_per_minute)
        return 7000 if self.llm_provider == "groq" else 0

    @property
    def effective_request_budget(self) -> int:
        if self.llm_max_request_tokens:
            return max(0, self.llm_max_request_tokens)
        return 5500 if self.llm_provider == "groq" else 0

    @property
    def hard_max_duration(self) -> float:
        return self.max_clip_duration * (1.0 + self.duration_tolerance)

    def ensure_dirs(self) -> None:
        for d in (
            self.input_dir,
            self.downloads_dir,
            self.transcripts_dir,
            self.highlights_dir,
            self.clips_dir,
            self.output_dir,
            self.logs_dir,
            self.logs_dir / "jobs",
            self.db_path.parent,
            self.fonts_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def public_dict(self) -> dict:
        """Settings without secrets (safe to store in the DB / reports)."""
        data = self.model_dump(mode="json")
        return {k: v for k, v in data.items() if not any(m in k for m in SECRET_MARKERS)}


def get_settings(**overrides) -> Settings:
    """Load settings from env/.env and apply non-None overrides."""
    clean = {k: v for k, v in overrides.items() if v is not None and v != ""}
    settings = Settings(**clean)
    settings.ensure_dirs()
    return settings
