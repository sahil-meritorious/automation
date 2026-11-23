"""Shorts Studio — Streamlit interface.  Launch with:  streamlit run app.py"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st
from pydantic import ValidationError

from config import (
    AUTO,
    BASE_DIR,
    CAPTION_STYLES,
    LAYOUTS,
    LLM_PROVIDERS,
    PRIVACY_OPTIONS,
    RESOLUTIONS,
    SECRET_MARKERS,
    TOGGLE_MODES,
    TOPIC_SOURCES,
    TRANSCRIPTION_MODELS,
    get_settings,
    resume_overrides,
)
from core import job_runner
from core.llm_client import PROVIDERS
from core.logging_setup import job_log_path, setup_logging
from database.database import get_database
from modules.downloader import VIDEO_EXTENSIONS, validate_source
from modules.video_finder import SOURCE_LABELS

st.set_page_config(page_title="Shorts Studio", page_icon="▶", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stTextInput, .stButton button { font-family: 'Inter', sans-serif; }
#MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }
h1, h2, h3, h4 { letter-spacing: -0.01em; font-weight: 600; }
.app-header { display: flex; align-items: baseline; gap: .75rem; margin-bottom: .25rem; }
.app-header .brand { font-size: 1.65rem; font-weight: 700; }
.app-header .tag { color: #8B93A7; font-size: .95rem; }
.section-label { text-transform: uppercase; font-size: .72rem; letter-spacing: .08em; color: #8B93A7;
                 font-weight: 600; margin: 1.4rem 0 .4rem; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: .72rem; font-weight: 500;
         margin: 0 4px 4px 0; border: 1px solid #2A3040; color: #C9CEDA; background: #161B26; }
.badge.ai { border-color: #3B3F8F; color: #C7C9FF; background: #1B1D3A; }
.badge.you { border-color: #2F5D46; color: #B9EBD0; background: #14271E; }
.status { display: inline-block; padding: 3px 10px; border-radius: 6px; font-size: .78rem; font-weight: 600; }
.status.ok { background: #13281D; color: #7EE2A8; } .status.warn { background: #2B2412; color: #F2C661; }
.status.err { background: #2E1518; color: #F28B8B; } .status.run { background: #1B1D3A; color: #A5A8FF; }
.status.idle { background: #1A1E28; color: #A9B0C0; }
.short-title { font-weight: 600; font-size: .98rem; line-height: 1.3; margin: .2rem 0 .15rem; }
.short-meta { color: #8B93A7; font-size: .8rem; margin-bottom: .35rem; }
.hook { font-size: .85rem; color: #E6E8EE; background: #161B26; border-left: 3px solid #6366F1;
        padding: 6px 10px; border-radius: 4px; margin: .35rem 0; }
.muted { color: #8B93A7; font-size: .85rem; }
div[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

STATUS = {
    "queued": ("Queued", "idle"), "running": ("Processing", "run"), "completed": ("Completed", "ok"),
    "completed_with_errors": ("Completed with issues", "warn"), "failed": ("Failed", "err"),
    "cancelled": ("Cancelled", "idle"), "interrupted": ("Interrupted", "warn"),
}
LAYOUT_LABELS = {
    AUTO: "Auto — AI decides per Short",
    "blur_fit": "Full frame on blurred background",
    "center_crop": "Centre crop (fills screen)",
    "offset_crop": "Custom crop position",
    "face_track": "Face / subject tracking",
}
STYLE_LABELS = {AUTO: "Auto — AI decides per Short", "dynamic": "Bold dynamic", "minimal": "Clean minimal",
                "karaoke": "Karaoke highlight"}
TOGGLE_LABELS = {"auto": "Auto — AI decides", "on": "Always on", "off": "Off"}
OVERLAY_LABELS = {AUTO: "Auto — AI hook line", "title": "Use the title", "off": "No on-screen text"}
CATEGORY_IDS = {"20": "Gaming", "24": "Entertainment", "22": "People & Blogs", "23": "Comedy", "17": "Sports",
                "28": "Science & Technology", "27": "Education"}

try:
    BASE = get_settings()
except ValidationError as exc:
    st.error(f"Your .env contains an invalid value:\n\n{exc}")
    st.stop()
setup_logging(BASE.logs_dir)
job_runner.startup_cleanup(BASE)
db = get_database(BASE.db_path)
ENV_PATH = BASE_DIR / ".env"


# ---------------------------------------------------------------- helpers
def save_env(values: dict) -> None:
    from dotenv import set_key

    ENV_PATH.touch(exist_ok=True)
    for k, v in values.items():
        if v is not None and v != "":
            set_key(str(ENV_PATH), k, str(v), quote_mode="never")


def open_folder(path: Path) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:
        st.warning(f"Could not open folder: {exc}")


@st.cache_data(show_spinner=False, max_entries=64)
def file_bytes(path: str, mtime: float) -> bytes:  # mtime busts the cache when a file is re-rendered
    return Path(path).read_bytes()


def tail(path: Path, lines: int = 60) -> str:
    if not path.exists():
        return "(no log yet)"
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        return "".join(fh.readlines()[-lines:])


def status_chip(status: str) -> str:
    label, cls = STATUS.get(status, (status, "idle"))
    return f'<span class="status {cls}">{label}</span>'


def label(text: str) -> None:
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def select(lbl: str, options, current, labels: dict, **kw):
    options = list(options)
    return st.selectbox(lbl, options, index=options.index(current) if current in options else 0,
                        format_func=lambda x: labels.get(x, x), **kw)


# ---------------------------------------------------------------- sidebar
def sidebar() -> dict:
    o: dict = {}
    with st.sidebar:
        st.markdown("### Settings")
        st.caption("Anything left on **Auto** is decided by the AI for each Short.")

        with st.expander("Creative direction", expanded=True):
            o["target_audience"] = st.text_area("Target audience", value=BASE.target_audience, height=96,
                                                help="The AI writes hooks, titles and hashtags for this audience")
            o["content_context"] = st.text_area("What the videos are about", value=BASE.content_context, height=68)
            o["caption_style"] = select("Caption style", (AUTO,) + CAPTION_STYLES, BASE.caption_style, STYLE_LABELS)
            o["layout"] = select("Vertical layout", (AUTO,) + LAYOUTS, BASE.layout, LAYOUT_LABELS)
            if o["layout"] == "offset_crop":
                o["crop_x_position"] = st.slider("Crop position (left to right)", 0.0, 1.0, BASE.crop_x_position)
            if o["layout"] in ("blur_fit", AUTO):
                o["foreground_zoom"] = st.slider("Gameplay zoom (full-frame layout)", 1.0, 1.8,
                                                 float(BASE.foreground_zoom), 0.05)
            o["punch_in_mode"] = select("Punch-in zoom at the hook", TOGGLE_MODES, BASE.punch_in_mode, TOGGLE_LABELS)
            o["zoom_mode"] = select("Slow zoom", TOGGLE_MODES, BASE.zoom_mode, TOGGLE_LABELS)
            o["overlay_text"] = select("On-screen text", (AUTO, "title", "off"), BASE.overlay_text, OVERLAY_LABELS)
            o["captions_enabled"] = st.checkbox("Animated captions", value=BASE.captions_enabled)
            if st.button("Save as default", key="save_creative"):
                save_env({"TARGET_AUDIENCE": o["target_audience"].replace("\n", " "),
                          "CONTENT_CONTEXT": o["content_context"].replace("\n", " "),
                          "CAPTION_STYLE": o["caption_style"], "LAYOUT": o["layout"],
                          "PUNCH_IN_MODE": o["punch_in_mode"], "ZOOM_MODE": o["zoom_mode"],
                          "OVERLAY_TEXT": o["overlay_text"]})
                st.success("Saved to .env")

        with st.expander("Selection"):
            o["num_shorts"] = st.slider("Maximum Shorts per video", 1, 25, BASE.num_shorts)
            c1, c2 = st.columns(2)
            o["min_clip_duration"] = c1.number_input("Min seconds", 5, 120, int(BASE.min_clip_duration))
            o["max_clip_duration"] = c2.number_input("Max seconds", 10, 180, int(BASE.max_clip_duration))
            o["min_score"] = st.slider("Quality threshold", 30, 95, int(BASE.min_score),
                                       help="Moments scoring below this are never exported")
            o["llm_rerank"] = st.checkbox("Global AI re-rank", value=BASE.llm_rerank)

        with st.expander("Output & audio"):
            res_names = list(RESOLUTIONS)
            cur_res = f"{BASE.output_width}x{BASE.output_height}"
            res = st.selectbox("Resolution", res_names, index=res_names.index(cur_res) if cur_res in res_names else 0)
            o["output_width"], o["output_height"] = RESOLUTIONS[res]
            o["caption_font"] = st.text_input("Caption font", value=BASE.caption_font,
                                              help="Any installed font, or .ttf files in assets/fonts")
            o["fade_transitions"] = st.checkbox("Fade in / out", value=BASE.fade_transitions)
            o["normalize_audio"] = st.checkbox("Loudness normalisation (-14 LUFS)", value=BASE.normalize_audio)
            o["background_music"] = st.text_input("Background music (rights-cleared only)",
                                                   value=BASE.background_music)
            if o["background_music"]:
                o["music_volume"] = st.slider("Music volume", 0.0, 0.5, BASE.music_volume, 0.01)

        with st.expander("Transcription"):
            o["transcription_model"] = select("Whisper model", TRANSCRIPTION_MODELS, BASE.transcription_model, {})
            o["transcription_language"] = st.text_input("Language code (blank = auto)",
                                                        value=BASE.transcription_language, max_chars=5)
            o["transcription_prompt"] = st.text_input("Spelling hints", value=BASE.transcription_prompt,
                                                      placeholder="GTA, Los Santos, Vice City, Lucia")

        with st.expander("Publishing"):
            o["youtube_privacy"] = select("Default privacy", PRIVACY_OPTIONS, BASE.youtube_privacy,
                                          {p: p.title() for p in PRIVACY_OPTIONS})
            o["youtube_category_id"] = select("Category", CATEGORY_IDS, BASE.youtube_category_id, CATEGORY_IDS)
            o["youtube_region"] = st.text_input("Trend region", value=BASE.youtube_region, max_chars=2)
            o["trend_research"] = st.checkbox("Use trending keywords", value=BASE.trend_research)
            o["trend_query"] = st.text_input("Trend topic (blank = automatic)", value=BASE.trend_query)
            o["youtube_upload_captions"] = st.checkbox("Upload subtitle track", value=BASE.youtube_upload_captions)
            o["youtube_description_footer"] = st.text_area("Description footer", value=BASE.youtube_description_footer,
                                                           height=68, placeholder="Credits or links")
            o["auto_upload"] = st.checkbox("Upload automatically when a job finishes", value=BASE.auto_upload)

        with st.expander("Accounts & API keys", expanded=not BASE.groq_api_key):
            groq = st.text_input("Groq API key", value=BASE.groq_api_key, type="password")
            if len(BASE.groq_keys) > 1:
                st.caption(f"{len(BASE.groq_keys)} Groq keys loaded from .env — rate-limited keys are skipped "
                           "automatically.")
            provider = select("AI provider", LLM_PROVIDERS, BASE.llm_provider, {})
            preset = PROVIDERS[provider]
            key_field = preset["key_field"]
            llm_key = ""
            if provider not in ("ollama", "groq"):
                llm_key = st.text_input(f"{provider.title()} API key", value=getattr(BASE, key_field, "") or "",
                                        type="password")
            default_model = BASE.llm_model if BASE.llm_provider == provider and BASE.llm_model else preset["model"]
            model = st.text_input("AI model", value=default_model)
            base_url = ""
            if provider in ("custom", "ollama"):
                base_url = st.text_input("Base URL", value=BASE.llm_base_url or preset["base_url"])
            yt_key = st.text_input("YouTube Data API key", value=BASE.youtube_api_key, type="password",
                                   help="Trend research and topic search")
            cid = st.text_input("YouTube OAuth client ID", value=BASE.youtube_client_id)
            csec = st.text_input("YouTube OAuth client secret", value=BASE.youtube_client_secret, type="password")
            o.update(groq_api_key=groq, llm_provider=provider, llm_model=model, llm_base_url=base_url,
                     youtube_api_key=yt_key, youtube_client_id=cid, youtube_client_secret=csec)
            if llm_key and key_field:
                o[key_field] = llm_key
            c1, c2 = st.columns(2)
            if c1.button("Save", key="save_keys"):
                env = {"GROQ_API_KEY": groq, "LLM_PROVIDER": provider, "LLM_MODEL": model,
                       "LLM_BASE_URL": base_url, "YOUTUBE_API_KEY": yt_key, "YOUTUBE_CLIENT_ID": cid,
                       "YOUTUBE_CLIENT_SECRET": csec, "YOUTUBE_PRIVACY": o["youtube_privacy"],
                       "YOUTUBE_CATEGORY_ID": o["youtube_category_id"], "YOUTUBE_REGION": o["youtube_region"]}
                if llm_key and key_field:
                    env[key_field.upper()] = llm_key
                save_env(env)
                st.success("Saved to .env")
            if c2.button("Test", key="test_keys"):
                from modules.transcription import check_groq_key

                for n, k in enumerate(get_settings(**o).groq_keys, 1):
                    ok, msg = check_groq_key(k)
                    (st.success if ok else st.error)(f"Groq key {n}: {msg}")
                try:
                    from core.llm_client import LLMClient

                    client = LLMClient(get_settings(**o))
                    client.ping()
                    st.success(f"AI OK: {client.label}")
                except Exception as exc:
                    st.error(f"AI: {exc}")
                if yt_key:
                    try:
                        from modules.trend_research import TrendResearcher

                        st.success(TrendResearcher(get_settings(**o)).check_key())
                    except Exception as exc:
                        st.error(f"YouTube API key: {exc}")
    return o


# ---------------------------------------------------------------- job views
@st.fragment(run_every=2)
def live_monitor(job_id: str) -> None:
    job = db.get_job(job_id)
    if not job:
        return
    progress = float(job.get("progress") or 0)
    st.progress(min(1.0, progress), text=f"{(job.get('stage') or '').title()} — {job.get('message') or ''}")
    with st.expander("Live log", expanded=False):
        st.code(tail(job_log_path(job_id, BASE.logs_dir), 40), language="log")
    if job["status"] not in ("queued", "running"):
        st.rerun()


def decision_badges(m: dict) -> str:
    by = m.get("decided_by") or {}
    items = [
        ("caption_style", STYLE_LABELS.get(m.get("caption_style"), m.get("caption_style", ""))),
        ("layout", LAYOUT_LABELS.get(m.get("layout"), m.get("layout") or "")),
        ("punch_in", "Punch-in" if m.get("punch_in") else "No punch-in"),
        ("zoom", "Slow zoom" if m.get("zoom") else "No slow zoom"),
    ]
    html = ""
    for key, text in items:
        if not text:
            continue
        who = by.get(key, "")
        cls = "ai" if who == "ai" else ("you" if who == "you" else "")
        suffix = " · AI" if who == "ai" else (" · you" if who == "you" else "")
        html += f'<span class="badge {cls}">{text}{suffix}</span>'
    return html


def shorts_tab(job: dict, report: dict) -> None:
    shorts = report["shorts"]
    if not shorts:
        st.info("No moment met the quality threshold. Lower the threshold or review the candidates tab.")
        return
    cols = st.columns(3, gap="large")
    for i, m in enumerate(shorts):
        with cols[i % 3]:
            with st.container(border=True):
                p = Path(m.get("output_path") or "")
                if m["status"] == "exported" and p.exists():
                    st.video(str(p))
                st.markdown(f'<div class="short-title">{m["rank"]}. {m["title"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="short-meta">Score {m["score"]:.0f} · {m["category"]} · '
                            f'{m["duration"]:.0f}s · {m["clip_start"]:.0f}s–{m["clip_end"]:.0f}s</div>',
                            unsafe_allow_html=True)
                if m.get("hook_text"):
                    st.markdown(f'<div class="hook">{m["hook_text"]}</div>', unsafe_allow_html=True)
                st.markdown(decision_badges(m), unsafe_allow_html=True)
                if m["status"] == "exported" and p.exists():
                    st.download_button("Download", data=file_bytes(str(p), p.stat().st_mtime), file_name=p.name,
                                       mime="video/mp4", key=f"dl_{job['id']}_{m['rank']}", width="stretch")
                else:
                    st.error(f"Not exported: {m.get('error', '')[:300]}")


def decisions_tab(report: dict) -> None:
    st.caption("Everything below was chosen by the AI unless marked “you”. Change any default in the sidebar.")
    for m in report["shorts"]:
        with st.container(border=True):
            c1, c2 = st.columns([2, 3])
            with c1:
                st.markdown(f"**{m['rank']}. {m['title']}**")
                st.markdown(decision_badges(m), unsafe_allow_html=True)
                st.markdown(f'<div class="muted">Hook at {m["hook_start"]:.1f}s · cut '
                            f'{m["clip_start"]:.1f}s → {m["clip_end"]:.1f}s</div>', unsafe_allow_html=True)
            with c2:
                if m.get("hook_text"):
                    st.markdown(f"**On-screen hook:** {m['hook_text']}")
                st.markdown(f"**Why it works:** {m.get('reason', '')}")
                if m.get("virality"):
                    st.markdown(f"**Viral trigger:** {m['virality']}")
                if m.get("edit_notes"):
                    st.markdown(f"**Edit:** {m['edit_notes']}")
                st.markdown(f"**Hashtags:** {' '.join(m.get('hashtags', []))}")
                if m.get("requirements"):
                    st.markdown("**Campaign requirements applied:** " + " · ".join(m["requirements"]))
                if m.get("tags"):
                    st.markdown(f'<div class="muted">Tags: {", ".join(m["tags"])}</div>', unsafe_allow_html=True)


def candidates_tab(report: dict) -> None:
    rows = [{
        "selected": f"#{c['rank']}" if c["selected"] else "",
        "final": round(c["final_score"] or 0),
        "first pass": round(c["score"]),
        "start": round(c["clip_start"], 1),
        "hook": round(c["hook_start"] or c["clip_start"], 1),
        "end": round(c["clip_end"], 1),
        "category": c["category"],
        "title": c["title"],
        "reason": c["reason"],
        "rejected because": c["reject_reason"],
    } for c in report["candidates"]]
    st.dataframe(rows, hide_index=True, width="stretch")


def render_results(job: dict) -> None:
    out_dir = Path(job["output_dir"]) if job.get("output_dir") else None
    report_path = out_dir / "report.json" if out_dir else None
    if not report_path or not report_path.exists():
        st.info("No results yet for this job.")
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summ = report["summary"]
    c = st.columns([1, 1, 1, 1, 1.3])
    c[0].metric("Candidates", summ["candidates"])
    c[1].metric("Selected", summ["selected"])
    c[2].metric("Exported", summ["exported"])
    c[3].metric("Failed", summ["failed"])
    if c[4].button("Open output folder", width="stretch"):
        open_folder(out_dir)

    t1, t2, t3, t4, t5 = st.tabs(["Shorts", "AI decisions", "Publish", "Candidates", "Report"])
    with t1:
        shorts_tab(job, report)
    with t2:
        decisions_tab(report)
    with t3:
        youtube_tab(job, out_dir)
    with t4:
        candidates_tab(report)
    with t5:
        md = out_dir / "report.md"
        if md.exists():
            st.markdown(md.read_text(encoding="utf-8"))
        for name, mime in (("metadata.csv", "text/csv"), ("metadata.json", "application/json")):
            f = out_dir / name
            if f.exists():
                st.download_button(f"Download {name}", f.read_bytes(), file_name=name, mime=mime,
                                   key=f"{name}_{job['id']}")


def youtube_settings():
    return get_settings(**sidebar_overrides)


def youtube_tab(job: dict, out_dir: Path) -> None:
    from modules.youtube_uploader import (
        YouTubeError, YouTubeUploader, auth_status, disconnect, pacific_day, save_edited_metadata, start_web_auth)

    s = youtube_settings()
    if "yt_status" not in st.session_state:
        st.session_state["yt_status"] = auth_status(s) if s.youtube_token_path.exists() else \
            {"connected": False, "message": "Not connected", "channel": None}
    status = st.session_state["yt_status"]

    with st.container(border=True):
        c1, c2, c3 = st.columns([3, 1, 1])
        if status["connected"]:
            ch = status["channel"]
            c1.markdown(f"{status_chip('completed').replace('Completed', 'Connected')} &nbsp; **{ch['title']}** · "
                        f"{ch['subscribers']} subscribers", unsafe_allow_html=True)
            if c3.button("Disconnect", key=f"yt_disc_{job['id']}", width="stretch"):
                disconnect(s)
                st.session_state.pop("yt_status", None)
                st.rerun()
        else:
            c1.markdown(f"{status_chip('cancelled').replace('Cancelled', 'Not connected')} &nbsp; "
                        f"<span class='muted'>{status['message']}</span>", unsafe_allow_html=True)
        if c2.button("Connect" if not status["connected"] else "Reconnect", key=f"yt_conn_{job['id']}",
                     width="stretch"):
            try:
                st.session_state["yt_auth_url"] = start_web_auth(s)
            except Exception as exc:
                st.error(str(exc))
        if st.session_state.get("yt_auth_url"):
            st.link_button("Sign in with Google", st.session_state["yt_auth_url"], type="primary")
            st.caption(f"Google returns you to {s.youtube_redirect_uri}. After approving, refresh the status.")
            if st.button("Refresh status", key=f"yt_ref_{job['id']}"):
                st.session_state.pop("yt_auth_url", None)
                st.session_state["yt_status"] = auth_status(s)
                st.rerun()
        used = db.quota_used(pacific_day())
        st.caption(f"API quota used today (estimate): {used:,} / {s.youtube_daily_quota:,} units · resets at "
                   f"midnight Pacific. Only publish content you own or have permission to republish.")

    metas = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8")) \
        if (out_dir / "metadata.json").exists() else []
    exported = [m for m in metas if m["status"] == "exported"]
    if not exported:
        st.info("No exported Shorts in this job.")
        return
    rows = {r["rank"]: r for r in db.list_shorts(job["id"])}

    label("Metadata — generated by AI, edit only if you want to")
    for m in exported:
        r = rows.get(m["rank"], {})
        vid = r.get("youtube_video_id")
        state = f" — {r.get('upload_status')}" if vid or r.get("upload_status") in ("failed", "pending") else ""
        with st.expander(f"{m['rank']}. {m['title']}{state}"):
            if vid:
                st.markdown(f"[Open on YouTube](https://youtube.com/shorts/{vid})")
            if r.get("upload_error"):
                st.caption(r["upload_error"][:300])
            k = f"{job['id']}_{m['rank']}"
            title = st.text_input("Title", m["title"], max_chars=100, key=f"yt_t_{k}")
            desc = st.text_area("Description", m.get("description", ""), height=100, key=f"yt_d_{k}")
            tags_h = st.text_input("Hashtags", " ".join(m.get("hashtags", [])), key=f"yt_h_{k}")
            tags = st.text_area("Tags", ", ".join(m.get("tags", [])), height=68, key=f"yt_g_{k}")
            if st.button("Save changes", key=f"yt_s_{k}"):
                save_edited_metadata(out_dir, m["rank"], title, desc, tags_h.split(),
                                     [x.strip() for x in tags.split(",") if x.strip()])
                st.success("Saved")
                st.rerun()

    label("Upload")
    with st.container(border=True):
        locked = YouTubeUploader.job_private_only(job)
        if locked:
            st.warning("Test project (source found with “Any public video”): uploads are forced to Private.")
        names = {m["rank"]: f"{m['rank']}. {m['title'][:60]}" for m in exported}
        not_up = [m["rank"] for m in exported if not rows.get(m["rank"], {}).get("youtube_video_id")]
        chosen = st.multiselect("Shorts", list(names), default=not_up, format_func=lambda r: names[r])
        c1, c2, c3 = st.columns(3)
        privacy = c1.selectbox(
            "Privacy",
            PRIVACY_OPTIONS,
            index=PRIVACY_OPTIONS.index(s.youtube_privacy),
            format_func=str.title,
            key=f"_priv_{job['id']}"
        )
        interval = c2.number_input("Hours between public Shorts (0 = now)", 0.0, 168.0,
                                   float(s.youtube_schedule_interval_hours), 1.0, key=f"yt_int_{job['id']}")
        force = c3.checkbox("Re-upload already uploaded", key=f"yt_force_{job['id']}")
        if privacy == "public":
            st.caption("A steady flow (one Short every 2–4 hours) usually performs better than posting all at once.")
        if st.button("Upload to YouTube", type="primary", disabled=not (chosen and status["connected"])):
            bar = st.progress(0.0, text="Starting upload…")
            try:
                results = YouTubeUploader(s, db).upload_job(
                    job["id"], ranks=chosen, privacy=privacy, interval_hours=interval, force=force,
                    progress=lambda f, msg: bar.progress(min(1.0, f), text=msg))
            except YouTubeError as exc:
                st.error(str(exc))
            else:
                bar.progress(1.0, text="Done")
                for r in results:
                    if r.get("url"):
                        when = f" — goes public {r['publish_at'][:16]}" if r.get("publish_at") else ""
                        st.success(f"{r['rank']}. {r['upload_status']}: {r['url']}{when}")
                    else:
                        st.error(f"{r['rank']}. {r['upload_status']}: {r.get('error', '')}")


def render_job(job_id: str) -> None:
    job = db.get_job(job_id)
    if not job:
        return
    c1, c2 = st.columns([4, 1])
    c1.markdown(f"{status_chip(job['status'])} &nbsp; <span class='muted'>{job['source']}</span>",
                unsafe_allow_html=True)

    try:
        job_cfg = json.loads(job.get("settings_json") or "{}")
    except ValueError:
        job_cfg = {}
    if job_cfg.get("campaign_json"):
        from modules.campaign_brief import CampaignBrief

        try:
            b = CampaignBrief.model_validate_json(job_cfg["campaign_json"])
            with st.expander(f"Campaign: {b.name or b.brand or 'requirements'}"):
                for line in b.requirement_lines():
                    st.markdown(f"- {line}")
                if job_cfg.get("logo_path"):
                    st.image(job_cfg["logo_path"], width=120)
        except Exception:
            pass

    if job["status"] in ("queued", "running"):
        if job_runner.is_running(job_id) and c2.button("Cancel", width="stretch"):
            job_runner.cancel_job(job_id)
            st.toast("Cancelling after the current step…")
        live_monitor(job_id)
        return

    if job["status"] in ("failed", "cancelled", "interrupted"):
        if job.get("error"):
            st.error(job["error"])
        st.caption("Completed stages are cached — resuming continues where it stopped.")
        if c2.button("Resume", type="primary", width="stretch"):
            saved = json.loads(job.get("settings_json") or "{}")
            job_runner.start_job(job["source"], get_settings(**resume_overrides(saved, sidebar_overrides)),
                                 job_id=job_id)
            st.rerun()
    else:
        st.caption(job.get("message") or "")
    render_results(job)
    with st.expander("Job log"):
        st.code(tail(job_log_path(job_id, BASE.logs_dir), 300), language="log")


# ---------------------------------------------------------------- OAuth callback
def handle_oauth_callback() -> None:
    """Google redirects to YOUTUBE_REDIRECT_URI (this app) with ?code=…&state=… or ?error=…"""
    qp = st.query_params
    if "code" not in qp and "error" not in qp:
        return
    from modules.youtube_uploader import auth_status, finish_web_auth, oauth_error_message

    s = youtube_settings()
    try:
        if "error" in qp:
            st.error(oauth_error_message(qp.get("error", ""), s))
        else:
            ch = finish_web_auth(s, qp.get("code", ""), qp.get("state", ""))
            st.success(f"YouTube connected: {ch['title']}. You can close any other tab of this app.")
            st.session_state["yt_status"] = auth_status(s)
            st.session_state.pop("yt_auth_url", None)
    except Exception as exc:
        st.error(str(exc))
    finally:
        st.query_params.clear()


# ---------------------------------------------------------------- new project
def start_from_topic(topic: str, count: int, source_mode: str) -> None:
    from core.llm_client import LLMClient
    from modules.video_finder import FinderError, VideoFinder, topic_job_overrides

    settings = get_settings(**{**sidebar_overrides, "topic_source": source_mode})
    try:
        llm = LLMClient(settings)
    except Exception:
        llm = None
    done = [r["video_id"] for r in db._query("SELECT video_id FROM videos")]
    with st.spinner(f"Finding the best source video for “{topic}”…"):
        try:
            found = VideoFinder(settings, llm).find(topic, count, exclude_ids=done)
        except FinderError as exc:
            st.error(str(exc))
            return
    first = None
    for v in found:
        job_settings = get_settings(**{**sidebar_overrides, **topic_job_overrides(settings, topic, v)})
        jid = job_runner.start_job(v.url, job_settings)
        first = first or jid
        st.session_state.setdefault("found_notes", {})[jid] = v.to_dict()
    st.session_state["job_id"] = first
    st.rerun()


def brief_summary(brief) -> None:
    """Show what the AI understood from a campaign brief."""
    with st.container(border=True):
        head = brief.name or "Campaign"
        st.markdown(f"**{head}**" + (f" · {brief.brand}" if brief.brand else ""))
        if brief.goal:
            st.caption(brief.goal)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Sources**")
            for u in brief.source_urls or ["—"]:
                st.markdown(f"- {u}")
            if brief.search_topic and not brief.source_urls:
                st.markdown(f"- search: {brief.search_topic}")
            if brief.segments:
                st.markdown("**Exact timings** (used as given)")
                st.dataframe([{"start": round(g.start, 1), "end": round(g.end, 1), "length": round(g.end - g.start, 1),
                               "video": g.url or "(the video)", "note": g.note} for g in brief.segments],
                             hide_index=True, width="stretch")
            else:
                st.markdown("**Timings:** none given — the AI finds the strongest hooks that match the brief")
        with c2:
            st.markdown("**Requirements**")
            lines = [l for l in brief.requirement_lines() if not l.startswith("Goal:")]
            for line in lines or ["No specific rules found"]:
                st.markdown(f"- {line}")
        for m in brief.missing_info:
            st.warning(m)


def campaign_panel() -> None:
    from core.llm_client import LLMClient
    from modules.campaign_brief import (
        LOGO_POSITIONS, BriefParser, CampaignBrief, campaign_overrides, save_campaign)
    from modules.video_finder import FinderError, VideoFinder

    text = st.text_area("Campaign requirements", height=220, label_visibility="collapsed",
                        placeholder="Paste the campaign brief: video links, timings, clip count and length, "
                                    "required hashtags / @tags / caption text, logo rules, dos and don'ts…")
    c1, c2, c3 = st.columns([3, 2, 1.4])
    extra_links = c1.text_area("Video links (optional)", height=80,
                               placeholder="Extra video links, one per line (if not in the brief)")
    logo = c2.file_uploader("Brand logo (optional)", type=["png", "jpg", "jpeg", "webp"])
    pos_options = ["brief"] + list(LOGO_POSITIONS)
    logo_pos = c3.selectbox("Logo position", pos_options,
                            format_func=lambda x: "As in brief" if x == "brief" else x.replace("_", " ").title())
    rights = st.checkbox("This campaign authorises me to clip and post the brand's content.")
    b1, b2, _ = st.columns([1.2, 1.6, 3])
    analyse = b1.button("Analyze requirements", disabled=not text.strip(), width="stretch")
    create = b2.button("Analyze & create clips", type="primary", disabled=not (text.strip() and rights),
                       width="stretch")

    if analyse or create:
        try:
            settings = get_settings(**sidebar_overrides)
        except ValidationError as exc:
            st.error(f"Invalid settings: {exc}")
            return
        cached = st.session_state.get("campaign_parsed")
        if cached and cached.get("text") == text:
            brief = CampaignBrief.model_validate(cached["brief"])
        else:
            try:
                llm = LLMClient(settings)
            except Exception:
                llm = None
            with st.spinner("Reading the campaign requirements…"):
                try:
                    brief = BriefParser(settings, llm).parse(text)
                except ValueError as exc:
                    st.error(str(exc))
                    return
            st.session_state["campaign_parsed"] = {"text": text, "brief": brief.model_dump()}
        for u in [x.strip() for x in extra_links.splitlines() if x.strip()]:
            ok, _ = validate_source(u)
            if ok and u not in brief.source_urls:
                brief.source_urls.append(u)
        if logo_pos != "brief":
            brief.logo_position = logo_pos
        if logo is not None:
            brief.logo_required = True
            brief.missing_info = [m for m in brief.missing_info if "logo" not in m.lower()]
        brief_summary(brief)
        if not create:
            return
        if not sidebar_overrides.get("groq_api_key"):
            st.error("Add your Groq API key under Settings → Accounts & API keys.")
            return
        sources = list(brief.source_urls)
        if not sources and brief.search_topic:
            try:
                llm = LLMClient(settings)
            except Exception:
                llm = None
            try:
                sources = [v.url for v in VideoFinder(settings, llm).find(brief.search_topic, 1)]
            except FinderError as exc:
                st.error(str(exc))
                return
        if not sources:
            st.error("No video to clip: add a link to the brief or the links box.")
            return
        saved = save_campaign(settings, brief, logo.getvalue() if logo is not None else None,
                              logo.name if logo is not None else "")
        first = None
        for src in sources:
            try:
                job_settings = get_settings(**{**sidebar_overrides,
                                               **campaign_overrides(settings, brief, saved["logo_path"], saved["slug"])})
            except ValidationError as exc:
                st.error(f"Campaign settings are invalid: {exc}")
                return
            jid = job_runner.start_job(src, job_settings)
            first = first or jid
        st.session_state.pop("campaign_parsed", None)
        st.session_state["job_id"] = first
        st.rerun()


def new_project() -> None:
    label("New project")
    with st.container(border=True):
        mode = st.segmented_control("Source", ["YouTube link", "Campaign", "Topic", "Local file"],
                                    default="YouTube link", label_visibility="collapsed") or "YouTube link"
        if mode == "Campaign":
            campaign_panel()
            return
        source, topic, count, topic_mode = "", "", 1, BASE.topic_source
        if mode == "YouTube link":
            source = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=…",
                                   label_visibility="collapsed")
            rights_text = "I own this video or have permission to edit and republish it."
        elif mode == "Local file":
            files = sorted(p for p in BASE.input_dir.glob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
            c1, c2 = st.columns([3, 2])
            source = c1.text_input("File path", placeholder="/home/you/Videos/gameplay.mp4",
                                   label_visibility="collapsed")
            if files:
                pick = c2.selectbox("From input folder", ["—"] + [p.name for p in files],
                                    label_visibility="collapsed")
                if pick != "—" and not source:
                    source = str(BASE.input_dir / pick)
            rights_text = "I own this video or have permission to edit and republish it."
        else:
            c1, c2, c3 = st.columns([3, 1, 2])
            topic = c1.text_input("Topic", placeholder="GTA 5 vs GTA 6 comparison, GTA 6 easter eggs…",
                                  label_visibility="collapsed")
            count = c2.number_input("Videos", 1, 3, 1, label_visibility="collapsed",
                                    help="How many source videos to use")
            topic_mode = c3.selectbox("Search in", TOPIC_SOURCES, index=TOPIC_SOURCES.index(BASE.topic_source),
                                      format_func=lambda x: SOURCE_LABELS[x], label_visibility="collapsed")
            if topic_mode == "any_public":
                st.caption("Testing mode: searches all public videos. Shorts from these projects can only be "
                           "uploaded as Private. Switch to “Channels I have permission from” and set "
                           "ALLOWED_CHANNEL_IDS in .env before publishing.")
            else:
                st.caption("The AI picks the source with the most viral potential and creates the Shorts. "
                           "Creative Commons sources are credited automatically; likely re-uploads are skipped.")
            rights_text = "I will only publish Shorts from sources I am allowed to reuse and will keep the credit."
        rights = st.checkbox(rights_text)
        ready = rights and bool(source.strip() if mode != "Topic" else topic.strip())
        if st.button("Create Shorts", type="primary", disabled=not ready):
            if not sidebar_overrides.get("groq_api_key"):
                st.error("Add your Groq API key under Settings → Accounts & API keys.")
                return
            try:
                settings = get_settings(**sidebar_overrides)
            except ValidationError as exc:
                st.error(f"Invalid settings: {exc}")
                return
            if settings.min_clip_duration >= settings.max_clip_duration:
                st.error("Minimum duration must be smaller than maximum duration.")
                return
            if mode == "Topic":
                start_from_topic(topic.strip(), int(count), topic_mode)
                return
            ok, msg = validate_source(source)
            if not ok:
                st.error(msg)
                return
            st.session_state["job_id"] = job_runner.start_job(source.strip(), settings)
            st.rerun()


# ---------------------------------------------------------------- page
sidebar_overrides = sidebar()
st.markdown('<div class="app-header"><span class="brand">Shorts Studio</span>'
            '<span class="tag">AI-directed vertical clips built for reach</span></div>', unsafe_allow_html=True)
handle_oauth_callback()
new_project()

jobs = db.list_jobs(40)
label("Projects")
if not jobs:
    st.info("No projects yet. Paste a link or enter a topic above. Tip: try a 5–10 minute video first.")
else:
    ids = [j["id"] for j in jobs]
    current = st.session_state.get("job_id")
    if current not in ids:
        current = ids[0]
    names = {j["id"]: f"{STATUS.get(j['status'], (j['status'],))[0]} · {j['created_at'][:16].replace('T', ' ')} · "
                      f"{(j['source'] or '')[:80]}" for j in jobs}
    chosen = st.selectbox("Project", ids, index=ids.index(current), format_func=lambda i: names[i],
                          label_visibility="collapsed")
    st.session_state["job_id"] = chosen
    note = st.session_state.get("found_notes", {}).get(chosen)
    if note:
        st.caption(f"Found for topic: “{note['title']}” by {note['channel']} · {note['duration'] / 60:.0f} min · "
                   f"{note['views']:,} views · licence {note['license']}. {note.get('reason', '')}")
    render_job(chosen)
