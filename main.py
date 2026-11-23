"""Command-line interface.

Examples
  python main.py self-test                       # verify FFmpeg + editing chain (no API keys needed)
  python main.py self-test --with-apis           # also verify Groq + LLM keys
  python main.py process "https://youtu.be/XXXXXXXXXXX" --confirm-rights
  python main.py process -f urls.txt --confirm-rights --shorts 5 --caption-style karaoke
  python main.py process "C:\\Videos\\my_gameplay.mp4" --confirm-rights
  python main.py jobs
  python main.py resume 20250101-120000-abc123
  python main.py youtube-auth                    # one-time: connect your YouTube channel
  python main.py trends "GTA 6"                  # show trending keywords for a topic
  python main.py upload JOB_ID --privacy public --interval-hours 4
  python main.py process URL --confirm-rights --upload   # process and upload in one go
  python main.py process --topic "GTA 5 vs GTA 6 comparison" --confirm-rights   # AI finds the video
  python main.py process --brief campaign.txt --logo logo.png --confirm-rights   # clipping campaign
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from config import (
    CAPTION_STYLES,
    LAYOUTS,
    LLM_PROVIDERS,
    PRIVACY_OPTIONS,
    RESOLUTIONS,
    TOPIC_SOURCES,
    TRANSCRIPTION_MODELS,
    get_settings,
)

RIGHTS_NOTICE = (
    "Only process videos you own or are authorised to edit and republish. "
    "Re-run with --confirm-rights to confirm this."
)


def _overrides(args: argparse.Namespace) -> dict:
    o = {
        "num_shorts": args.shorts,
        "min_clip_duration": args.min_duration,
        "max_clip_duration": args.max_duration,
        "caption_style": args.caption_style,
        "layout": args.layout,
        "llm_provider": args.provider,
        "llm_model": args.model,
        "transcription_model": args.transcription_model,
        "transcription_language": args.language,
        "content_context": args.context,
        "min_score": args.min_score,
        "background_music": args.music,
    }
    if args.resolution:
        o["output_width"], o["output_height"] = RESOLUTIONS[args.resolution]
    if args.no_rerank:
        o["llm_rerank"] = False
    if args.no_captions:
        o["captions_enabled"] = False
    if args.keep_intermediate:
        o["keep_intermediate"] = True
    if args.upload:
        o["auto_upload"] = True
    if args.privacy:
        o["youtube_privacy"] = args.privacy
    if args.interval_hours is not None:
        o["youtube_schedule_interval_hours"] = args.interval_hours
    if args.no_trends:
        o["trend_research"] = False
    if args.topic_source:
        o["topic_source"] = args.topic_source
    return o


def _print_result(res: dict) -> None:
    print()
    print("=" * 70)
    print(f"Job {res['job_id']}: {res['status'].upper()}  —  {res['source']}")
    if res.get("error"):
        print(f"Error: {res['error']}")
    shorts = res.get("shorts") or []
    for m in shorts:
        mark = "OK " if m["status"] == "exported" else "ERR"
        print(f"  [{mark}] #{m['rank']:>2} score {m['score']:>5.1f}  {m['duration']:>5.1f}s  {m['title']}")
        if m["status"] != "exported":
            print(f"        {m.get('error', '')[:200]}")
    if res.get("output_dir"):
        print(f"Output folder: {res['output_dir']}")
    print("=" * 70)


def cmd_process(args: argparse.Namespace) -> int:
    from core.pipeline import ShortsPipeline
    from modules.downloader import validate_source

    if not args.confirm_rights:
        print(RIGHTS_NOTICE)
        return 2
    sources: List[str] = list(args.sources or [])
    if args.file:
        for line in Path(args.file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                sources.append(line)
    if not sources and not args.topic and not args.brief:
        print("No sources given. Pass URLs/paths, --file urls.txt, --topic \"GTA 6 easter eggs\" or --brief brief.txt")
        return 2
    valid = []
    for src in sources:
        ok, msg = validate_source(src)
        print(f"{'✓' if ok else '✗'} {src} — {msg}")
        if ok:
            valid.append(src)

    settings = get_settings(**_overrides(args))
    jobs = [(src, settings) for src in valid]
    if args.topic:
        from core.llm_client import LLMClient
        from database.database import get_database
        from modules.video_finder import FinderError, VideoFinder, topic_job_overrides

        db = get_database(settings.db_path)
        done = [r["video_id"] for r in db._query("SELECT video_id FROM videos")]
        try:
            llm = LLMClient(settings)
        except Exception:
            llm = None  # ranking by relevance/views still works
        try:
            found = VideoFinder(settings, llm).find(args.topic, args.videos, exclude_ids=done)
        except FinderError as exc:
            print(f"✗ {exc}")
            return 1
        for v in found:
            print(f"✓ Topic '{args.topic}' → {v.title} ({v.channel}, {v.duration / 60:.0f} min) {v.url}")
            jobs.append((v.url, get_settings(**{**_overrides(args), **topic_job_overrides(settings, args.topic, v)})))
    if args.brief:
        from core.llm_client import LLMClient
        from modules.campaign_brief import BriefParser, campaign_overrides, save_campaign

        try:
            llm = LLMClient(settings)
        except Exception:
            llm = None
        brief = BriefParser(settings, llm).parse(Path(args.brief).read_text(encoding="utf-8"))
        if args.logo_position:
            brief.logo_position = args.logo_position
        logo_bytes = Path(args.logo).read_bytes() if args.logo else None
        saved = save_campaign(settings, brief, logo_bytes, args.logo or "")
        print(f"Campaign: {brief.name or brief.brand or 'unnamed'}  (saved to {saved['dir']})")
        for line in brief.requirement_lines():
            print(f"  • {line}")
        for m in brief.missing_info:
            print(f"  ! {m}")
        brief_sources = list(brief.source_urls) or list(valid)
        if not brief_sources:
            print("✗ The brief has no video link — pass one as an argument too.")
            return 2
        over = campaign_overrides(settings, brief, saved["logo_path"], saved["slug"])
        jobs = [(src, get_settings(**{**_overrides(args), **over})) for src in brief_sources]
    if not jobs:
        return 2
    if settings.auto_upload:
        from modules.youtube_uploader import auth_status

        st = auth_status(settings)
        if not st["connected"]:
            print(f"--upload requested but YouTube is not ready: {st['message']}")
            return 2
        print(f"YouTube: {st['message']} — uploads will be {settings.youtube_privacy}")
    cancel = threading.Event()

    def run_one(job: tuple) -> dict:
        return ShortsPipeline(job[1], cancel_event=cancel).run(job[0])

    results = []
    try:
        if args.parallel > 1 and len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=args.parallel) as pool:
                results = list(pool.map(run_one, jobs))
        else:
            for job in jobs:
                results.append(run_one(job))
    except KeyboardInterrupt:
        print("\nInterrupted — finishing current step. Completed work is cached; run again to resume.")
        cancel.set()
        return 130
    for r in results:
        _print_result(r)
    return 0 if all(r["status"].startswith("completed") for r in results) else 1


def cmd_resume(args: argparse.Namespace) -> int:
    from core.pipeline import ShortsPipeline
    from database.database import get_database

    base = get_settings()
    job = get_database(base.db_path).get_job(args.job_id)
    if not job:
        print(f"Job {args.job_id} not found")
        return 2
    from config import resume_overrides

    saved = json.loads(job.get("settings_json") or "{}")
    settings = get_settings(**resume_overrides(saved))  # AI provider/model come from the current .env
    res = ShortsPipeline(settings).run(job["source"], job_id=job["id"])
    _print_result(res)
    return 0 if res["status"].startswith("completed") else 1


def cmd_jobs(args: argparse.Namespace) -> int:
    from database.database import get_database

    s = get_settings()
    jobs = get_database(s.db_path).list_jobs(args.limit)
    if not jobs:
        print("No jobs yet.")
    for j in jobs:
        print(f"{j['id']}  {j['status']:<22} {float(j['progress'] or 0) * 100:5.1f}%  {j['source'][:60]}")
        if j.get("message"):
            print(f"    {j['message'][:120]}")
    return 0


def cmd_youtube_auth(args: argparse.Namespace) -> int:
    from modules.youtube_uploader import YouTubeError, authorize

    s = get_settings()
    try:
        ch = authorize(s, open_browser=not args.no_browser)
    except YouTubeError as exc:
        print(f"✗ {exc}")
        return 1
    print(f"✓ Connected to channel '{ch['title']}' ({ch['subscribers']} subscribers) — {ch['url']}")
    print(f"  Token saved to {s.youtube_token_path}")
    return 0


def cmd_youtube_status(args: argparse.Namespace) -> int:
    from database.database import get_database
    from modules.trend_research import TrendResearcher
    from modules.youtube_uploader import auth_status, pacific_day

    s = get_settings()
    st = auth_status(s)
    print(f"{'✓' if st['connected'] else '✗'} Upload account: {st['message']}")
    if s.youtube_api_key:
        try:
            print(f"✓ Trend research: {TrendResearcher(s).check_key()}")
        except Exception as exc:
            print(f"✗ Trend research: {exc}")
    else:
        print("• Trend research: YOUTUBE_API_KEY not set (optional)")
    used = get_database(s.db_path).quota_used(pacific_day())
    print(f"• Estimated quota used today (Pacific): {used} / {s.youtube_daily_quota} units")
    return 0 if st["connected"] else 1


def cmd_upload(args: argparse.Namespace) -> int:
    from datetime import datetime

    from modules.youtube_uploader import YouTubeError, YouTubeUploader

    s = get_settings()
    start = None
    if args.schedule_start:
        try:
            start = datetime.fromisoformat(args.schedule_start)
        except ValueError:
            print("--schedule-start must look like 2026-11-20T18:00 (local time) or include an offset")
            return 2
    ranks = [int(x) for x in args.ranks.split(",")] if args.ranks else None

    def prog(f: float, m: str) -> None:
        print(f"\r[{f * 100:5.1f}%] {m[:90]:<90}", end="", flush=True)

    try:
        results = YouTubeUploader(s).upload_job(args.job_id, ranks=ranks, privacy=args.privacy,
                                                schedule_start=start, interval_hours=args.interval_hours,
                                                force=args.force, progress=prog)
    except YouTubeError as exc:
        print(f"\n✗ {exc}")
        return 1
    print()
    for r in results:
        when = f" → public at {r['publish_at']}" if r.get("publish_at") else ""
        line = r.get("url") or r.get("error", "")
        print(f"  #{r['rank']:>2} {r['upload_status']:<10} {line}{when}  {r['title'][:50]}")
    return 0 if all(r["upload_status"] in ("uploaded", "scheduled") for r in results) else 1


def cmd_trends(args: argparse.Namespace) -> int:
    from modules.trend_research import TrendError, TrendResearcher

    s = get_settings(**({"youtube_region": args.region} if args.region else {}))
    try:
        t = TrendResearcher(s).research(args.query, force=args.refresh)
    except TrendError as exc:
        print(f"✗ {exc}")
        return 1
    print(f"Topic '{t['query']}' — {t['videos_analysed']} recent Shorts analysed ({t['region']})\n")
    print("Keywords: " + ", ".join(k["term"] for k in t["keywords"][:40]))
    print("Hashtags: " + " ".join(t["hashtags"][:15]))
    print("\nTop recent Shorts:")
    for v in t["top_titles"][:10]:
        print(f"  {v['views']:>12,}  {v['title']}")
    return 0


def cmd_self_test(args: argparse.Namespace) -> int:
    """Generate a synthetic clip and run the real editing chain end-to-end."""
    from core.logging_setup import setup_logging
    from core.models import Word
    from core.utils import check_binaries, configure_binaries, run_ffmpeg
    from modules.caption_generator import CaptionGenerator, caption_spans
    from modules.quality_checker import QualityChecker
    from modules.vertical_editor import VerticalEditor

    s = get_settings()
    if s.layout == "auto":
        s = get_settings(layout="blur_fit")
    setup_logging(s.logs_dir)
    configure_binaries(s.ffmpeg_path, s.ffprobe_path)
    ok = True
    print("1) FFmpeg")
    try:
        for k, v in check_binaries().items():
            print(f"   ✓ {k}: {v}")
    except Exception as exc:
        print(f"   ✗ {exc}")
        return 1

    work = s.clips_dir / "_selftest"
    work.mkdir(parents=True, exist_ok=True)
    src = work / "raw.mp4"
    print("2) Generating a 20 s 1280x720 test clip with a tone")
    run_ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=20",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=20",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(src)])
    print("   ✓ test clip created")

    text = ("wait what just happened there that car flew straight into the air "
            "and the cops did not even notice this game is absolutely insane").split()
    words, t = [], 1.0
    for w in text:
        words.append(Word(word=w, start=t, end=t + 0.35))
        t += 0.45
    styles = [args.style] if args.style else list(CAPTION_STYLES)
    for style in styles:
        print(f"3) Rendering a 9:16 Short — caption style '{style}', layout '{s.layout}'")
        editor = VerticalEditor(s)
        plan = editor.plan(src, hook_rel=3.0, duration=20.0, keyword_times=[6.0])
        ass_text = CaptionGenerator(style, s.caption_font, s.caption_font_size, plan.caption_y).build(
            words, 20.0, keywords=["insane"], title="Self-test: the car flew away", title_y=plan.title_y)
        ass = work / "captions.ass"
        ass.write_text(ass_text, encoding="utf-8")
        out = s.output_dir / "_selftest" / f"selftest_{style}.mp4"
        try:
            editor.render(src, out, plan, ass, mode="full")
        except Exception as exc:
            print(f"   ✗ render failed: {exc}")
            ok = False
            continue
        qc = QualityChecker(s.output_width, s.output_height, 15, 60).check(out, 20.0, caption_spans(ass_text), True)
        status = "✓" if qc.passed else "✗"
        print(f"   {status} {out}  {qc.info.get('width')}x{qc.info.get('height')} "
              f"{qc.info.get('duration', 0):.1f}s {qc.info.get('size_mb')} MB")
        for e in qc.errors:
            print(f"     error: {e}")
        for w in qc.warnings:
            print(f"     warning: {w}")
        ok = ok and qc.passed

    if args.with_apis:
        print("4) API keys")
        from modules.transcription import check_groq_key

        for n, key in enumerate(s.groq_keys or [""], 1):
            g_ok, g_msg = check_groq_key(key)
            print(f"   {'✓' if g_ok else '✗'} Groq key {n}: {g_msg}")
            ok = ok and g_ok
        try:
            from core.llm_client import LLMClient

            llm = LLMClient(s)
            print(f"   ✓ LLM {llm.label}: {llm.ping()}")
        except Exception as exc:
            print(f"   ✗ LLM: {exc}")
            ok = False

    print("\nSELF-TEST " + ("PASSED" if ok else "FAILED"))
    if ok:
        print(f"Open the test Shorts in: {s.output_dir / '_selftest'}")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="AI YouTube Shorts generator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("process", help="process one or more videos")
    p.add_argument("sources", nargs="*", help="YouTube URLs or local video paths")
    p.add_argument("-f", "--file", help="text file with one URL/path per line")
    p.add_argument("--confirm-rights", action="store_true", help="confirm you are authorised to use these videos")
    p.add_argument("--shorts", type=int, help="max number of Shorts per video (default 10)")
    p.add_argument("--min-duration", type=float)
    p.add_argument("--max-duration", type=float)
    p.add_argument("--min-score", type=float, help="quality threshold 0-100 (default 60)")
    p.add_argument("--caption-style", choices=CAPTION_STYLES + ("auto",), help="default auto (AI decides)")
    p.add_argument("--layout", choices=LAYOUTS + ("auto",), help="default auto (AI decides)")
    p.add_argument("--resolution", choices=list(RESOLUTIONS))
    p.add_argument("--provider", choices=LLM_PROVIDERS)
    p.add_argument("--model", help="LLM model name")
    p.add_argument("--transcription-model", choices=TRANSCRIPTION_MODELS)
    p.add_argument("--language", help="ISO-639-1 language code, e.g. en (default auto)")
    p.add_argument("--context", help="what the video is about, e.g. 'GTA 6 gameplay commentary'")
    p.add_argument("--music", help="background music file/folder you have rights to use")
    p.add_argument("--no-rerank", action="store_true", help="skip the global LLM re-rank pass")
    p.add_argument("--no-captions", action="store_true")
    p.add_argument("--keep-intermediate", action="store_true")
    p.add_argument("--parallel", type=int, default=1, help="videos processed in parallel (default 1)")
    p.add_argument("--upload", action="store_true", help="upload exported Shorts to YouTube when done")
    p.add_argument("--privacy", choices=PRIVACY_OPTIONS, help="YouTube privacy for --upload")
    p.add_argument("--interval-hours", type=float, help="schedule public Shorts N hours apart")
    p.add_argument("--no-trends", action="store_true", help="skip trending-keyword research")
    p.add_argument("--brief", help="campaign brief text file: the AI follows its requirements")
    p.add_argument("--logo", help="brand logo image (png/jpg/webp) for --brief")
    p.add_argument("--logo-position", choices=("top_right", "top_left", "top_center", "bottom_right", "bottom_left"))
    p.add_argument("--topic", help='find source video(s) for a topic, e.g. "GTA 5 vs GTA 6 comparison"')
    p.add_argument("--videos", type=int, default=1, help="how many videos to use for --topic (default 1)")
    p.add_argument("--topic-source", choices=TOPIC_SOURCES, help="where --topic may search (default creative_commons)")
    p.set_defaults(func=cmd_process)

    ya = sub.add_parser("youtube-auth", help="connect your YouTube channel (one-time browser consent)")
    ya.add_argument("--no-browser", action="store_true", help="print the URL instead of opening a browser")
    ya.set_defaults(func=cmd_youtube_auth)

    ys = sub.add_parser("youtube-status", help="check YouTube connection, API key and quota")
    ys.set_defaults(func=cmd_youtube_status)

    u = sub.add_parser("upload", help="upload a finished job's Shorts to YouTube")
    u.add_argument("job_id")
    u.add_argument("--privacy", choices=PRIVACY_OPTIONS, help="default from YOUTUBE_PRIVACY (private)")
    u.add_argument("--ranks", help="only these Shorts, e.g. 1,2,5")
    u.add_argument("--interval-hours", type=float, help="public only: publish one Short every N hours")
    u.add_argument("--schedule-start", help="public only: first publish time, e.g. 2026-11-20T18:00")
    u.add_argument("--force", action="store_true", help="re-upload Shorts that are already on YouTube")
    u.set_defaults(func=cmd_upload)

    tr = sub.add_parser("trends", help="show trending keywords for a topic (needs YOUTUBE_API_KEY)")
    tr.add_argument("query")
    tr.add_argument("--region", help="2-letter region, e.g. US, IN, GB")
    tr.add_argument("--refresh", action="store_true", help="ignore today's cache")
    tr.set_defaults(func=cmd_trends)

    r = sub.add_parser("resume", help="resume an interrupted/failed job")
    r.add_argument("job_id")
    r.set_defaults(func=cmd_resume)

    j = sub.add_parser("jobs", help="list recent jobs")
    j.add_argument("--limit", type=int, default=20)
    j.set_defaults(func=cmd_jobs)

    t = sub.add_parser("self-test", help="verify FFmpeg and the editing pipeline")
    t.add_argument("--with-apis", action="store_true", help="also check Groq and LLM API keys")
    t.add_argument("--style", choices=CAPTION_STYLES, help="test only one caption style")
    t.set_defaults(func=cmd_self_test)
    return ap


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # ✓/✗ on Windows consoles
        except Exception:
            pass
    args = build_parser().parse_args()
    t0 = time.time()
    code = args.func(args)
    if args.cmd == "process":
        print(f"Total time: {(time.time() - t0) / 60:.1f} min")
    return code


if __name__ == "__main__":
    sys.exit(main())
