"""End-to-end orchestration with checkpointing.

Every stage persists its result to disk (download, audio, chunks, per-chunk
transcripts, merged transcript, per-window highlight analysis, ranking,
metadata, rendered Shorts). Re-running the same source — or resuming a job —
skips completed work, so a failure late in the pipeline never loses the
expensive transcription/analysis steps.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import Settings
from core.llm_client import LLMClient, fill_template, load_prompt
from core.logging_setup import close_job_logger, get_job_logger, setup_logging
from core.models import ClipCandidate, ShortMetadata, Transcript, VideoInfo
from core.utils import (
    JobCancelled,
    atomic_write_json,
    atomic_write_text,
    configure_binaries,
    check_binaries,
    ensure_disk_space,
    fingerprint,
    read_json,
    remove_path,
    slugify,
)
from database.database import Database, get_database
from modules.audio_chunker import AudioChunker
from modules.audio_extractor import extract_audio
from modules.caption_generator import CaptionGenerator, build_srt, caption_spans
from modules.clip_extractor import extract_clip
from modules.downloader import Downloader
from modules.campaign_brief import brief_prompt, load_brief
from modules.highlight_detector import HighlightDetector, format_transcript_window
from modules.highlight_ranker import HighlightRanker, is_duplicate
from modules.metadata_generator import MetadataGenerator, write_metadata_files, write_report
from modules.quality_checker import QualityChecker
from modules.transcript_merger import merge_chunk_results
from modules.transcription import GroqTranscriber
from modules.vertical_editor import VerticalEditor

ProgressCallback = Callable[[str, float, str], None]

# (stage, start %, end %) of overall progress
STAGES: Dict[str, Tuple[float, float]] = {
    "download": (0.00, 0.12),
    "audio": (0.12, 0.16),
    "chunking": (0.16, 0.18),
    "transcription": (0.18, 0.50),
    "highlights": (0.50, 0.66),
    "ranking": (0.66, 0.70),
    "metadata": (0.70, 0.72),
    "rendering": (0.72, 0.98),
    "report": (0.98, 1.00),
    "upload": (1.00, 1.00),
}


class ShortsPipeline:
    def __init__(
        self,
        settings: Settings,
        db: Optional[Database] = None,
        progress: Optional[ProgressCallback] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        self.s = settings
        settings.ensure_dirs()
        setup_logging(settings.logs_dir)
        configure_binaries(settings.ffmpeg_path, settings.ffprobe_path)
        self.db = db or get_database(settings.db_path)
        self.progress_cb = progress
        self.cancel_event = cancel_event or threading.Event()
        self.job_id = ""
        self.log = logging.getLogger("shorts.pipeline")
        self._last_db_update = 0.0

    # ------------------------------------------------------------ helpers
    def _check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise JobCancelled("Job cancelled by user")

    def _progress(self, stage: str, frac: float, message: str, force: bool = False) -> None:
        lo, hi = STAGES.get(stage, (0.0, 1.0))
        overall = lo + (hi - lo) * max(0.0, min(1.0, frac))
        now = time.time()
        if force or now - self._last_db_update > 0.7:
            self._last_db_update = now
            self.db.update_job(self.job_id, stage=stage, progress=round(overall, 4), message=message[:300])
        if self.progress_cb:
            try:
                self.progress_cb(stage, overall, message)
            except Exception:  # never let UI callbacks break the pipeline
                pass

    def _stage(self, stage: str, message: str) -> Callable[[float, str], None]:
        self._check_cancel()
        self.log.info("=== %s: %s", stage.upper(), message)
        self._progress(stage, 0.0, message, force=True)
        return lambda f, m: (self._check_cancel(), self._progress(stage, f, m))[-1]

    # ------------------------------------------------------------ main entry
    def run(self, source: str, job_id: Optional[str] = None) -> Dict:
        """Process one video. Returns a summary dict (never raises for pipeline errors)."""
        self.job_id = job_id or self.db.create_job(source, self.s.public_dict())
        self.log = get_job_logger(self.job_id, self.s.logs_dir)
        self.db.update_job(self.job_id, status="running", error=None, message="Starting",
                           settings_json=json.dumps(self.s.public_dict(), default=str))
        result: Dict = {"job_id": self.job_id, "source": source, "status": "running", "shorts": [], "error": None}
        t0 = time.time()
        try:
            check_binaries()
            outcome = self._run(source)
            result.update(outcome)
            failed = sum(1 for m in outcome["shorts"] if m["status"] != "exported")
            status = "completed_with_errors" if failed else "completed"
            msg = (f"Done: {len(outcome['shorts']) - failed} Short(s) exported"
                   + (f", {failed} failed" if failed else "")
                   + f" in {(time.time() - t0) / 60:.1f} min")
            if not outcome["shorts"]:
                msg = "Done: no moments met the quality bar (try lowering MIN_SCORE)"
            result["status"] = status
            self.db.update_job(self.job_id, status=status, progress=1.0, stage="done", message=msg)
            self.log.info(msg)
        except JobCancelled:
            result.update(status="cancelled", error="Cancelled by user")
            self.db.update_job(self.job_id, status="cancelled", message="Cancelled — completed work is kept; resume any time")
            self.log.warning("Job cancelled. Completed stages are cached and will be reused on resume.")
        except Exception as exc:
            result.update(status="failed", error=str(exc))
            self.db.update_job(self.job_id, status="failed", error=str(exc)[:2000],
                               message=f"Failed: {str(exc).splitlines()[0][:200]}")
            self.log.exception("Job failed: %s", exc)
        finally:
            close_job_logger(self.log)
        return result

    # ------------------------------------------------------------ stages
    def _run(self, source: str) -> Dict:
        s = self.s
        # 1. Acquire the video ------------------------------------------------
        p = self._stage("download", "Validating input and retrieving video")
        info: VideoInfo = Downloader(s, self.log, self.cancel_event).resolve(source, p)
        self.db.upsert_video(info.model_dump())
        self.db.update_job(self.job_id, video_id=info.video_id)
        self.log.info("Video: '%s' | %s | %.0fs | %s", info.title, info.video_id, info.duration, info.resolution)
        if not info.has_audio:
            raise RuntimeError("The video has no audio track, so it cannot be transcribed.")
        if info.duration < s.min_clip_duration:
            raise RuntimeError(f"Video is only {info.duration:.0f}s long — shorter than the minimum clip length.")

        work = s.transcripts_dir / info.video_id
        hl_dir = s.highlights_dir / info.video_id
        out_root = s.output_dir / s.campaign_slug if s.campaign_slug else s.output_dir
        out_dir = out_root / f"{info.video_id}_{slugify(info.title, 40)}"
        for d in (work, hl_dir, out_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.db.update_job(self.job_id, output_dir=str(out_dir))

        transcript = self._transcribe(info, work)

        llm = LLMClient(s, self.log)
        brief = load_brief(s)
        segments = brief.segments_for(info.video_id, source) if brief else []
        if brief:
            self.log.info("Campaign brief: %s", brief.name or brief.brand or "unnamed")
            for line in brief.requirement_lines():
                self.log.info("  • %s", line)
        if segments:
            # 5/6. Exact timings from the campaign brief ---------------------------
            self._stage("highlights", f"Using {len(segments)} segment(s) specified by the campaign")
            selected = self._direct_segments(segments, transcript, info, llm, hl_dir)
            all_cands = list(selected)
            if brief.extra_clips:
                p = self._stage("highlights", f"Finding up to {brief.extra_clips} extra clip(s) that fit the brief")
                extra_s = s.model_copy(update={"num_shorts": brief.extra_clips + len(selected)})
                found = HighlightDetector(extra_s, llm, self.log, self.cancel_event).detect(transcript, info, hl_dir, p)
                ranker = HighlightRanker(extra_s, llm if s.llm_rerank else None, self.log, self.cancel_event)
                more, cands = ranker.rank(found, transcript, info, hl_dir)
                added = 0
                for c in more:
                    if added >= brief.extra_clips:
                        break
                    if any(is_duplicate(c, g) for g in selected):
                        c.selected, c.reject_reason = False, "overlaps a segment given by the campaign"
                        continue
                    added += 1
                    c.rank = len(selected) + 1
                    selected.append(c)
                for c in cands:
                    if c.selected and c not in selected:
                        c.selected = False
                        c.reject_reason = c.reject_reason or "not needed (campaign limit reached)"
                all_cands += cands
                self.log.info("Added %d AI-found clip(s) to the %d campaign segment(s)", added, len(segments))
            durations = [c.duration for c in selected if c.id.startswith("g")]
            self.s = s = s.model_copy(update={  # QC must accept the brand's exact lengths
                "min_clip_duration": min([s.min_clip_duration] + durations) - 0.5,
                "max_clip_duration": max([s.max_clip_duration] + durations) + 0.5, "duration_tolerance": 0.0})
        else:
            # 5. Highlight detection ----------------------------------------------
            p = self._stage("highlights", "Analysing transcript for highlight moments")
            candidates = HighlightDetector(s, llm, self.log, self.cancel_event).detect(transcript, info, hl_dir, p)
            self.log.info("Detection produced %d candidate(s)", len(candidates))

            # 6. Ranking ------------------------------------------------------------
            self._stage("ranking", f"Ranking {len(candidates)} candidates")
            ranker = HighlightRanker(s, llm if s.llm_rerank else None, self.log, self.cancel_event)
            selected, all_cands = ranker.rank(candidates, transcript, info, hl_dir)
        for c in selected:
            self.log.info("  #%d %.0f [%s] %.0f–%.0fs (%.0fs) %s", c.rank, c.final_score, c.category,
                          c.clip_start, c.clip_end, c.duration, c.title)

        # 7. Metadata -----------------------------------------------------------
        self._stage("metadata", "Generating titles, descriptions and hashtags")
        metas = MetadataGenerator(s, llm, self.log, self.cancel_event).generate(selected, transcript, info, hl_dir)

        # 8. Rendering ----------------------------------------------------------
        metas = self._render_all(metas, info, transcript, out_dir)

        # 9. Reports ------------------------------------------------------------
        self._stage("report", "Writing metadata and report")
        write_metadata_files(out_dir, metas)
        write_report(out_dir, info, transcript, all_cands, metas, s, self.job_id, llm.label)
        for m in metas:
            self.db.save_short(self.job_id, info.video_id, m.model_dump())
        self._cleanup(info, work)
        if s.auto_upload and any(m.status == "exported" for m in metas):
            self._auto_upload()
        return {"video": info.model_dump(), "output_dir": str(out_dir),
                "shorts": [m.model_dump() for m in metas], "candidates": len(all_cands)}

    def _transcribe(self, info: VideoInfo, work: Path) -> Transcript:
        s = self.s
        fp = fingerprint([info.duration, s.transcription_model, s.transcription_language, s.transcription_prompt,
                          s.audio_chunk_seconds, s.audio_chunk_overlap])
        transcript_path = work / "transcript.json"
        cached = read_json(transcript_path)
        if cached and cached.get("fingerprint") == fp:
            self._progress("transcription", 1.0, "Using cached transcript", force=True)
            self.log.info("Using cached transcript (%d segments)", len(cached.get("transcript", [])))
            return Transcript.model_validate(cached)

        # 2. Audio
        self._stage("audio", "Extracting audio")
        wav = extract_audio(Path(info.source_path), work / "audio_16k.wav", info.duration, s.min_free_disk_gb, self.log)

        # 3. Chunking
        self._stage("chunking", "Splitting audio into chunks")
        chunks = AudioChunker(s, self.log).create_chunks(wav, info.duration, work / "audio_chunks")

        # 4. Transcription
        p = self._stage("transcription", f"Transcribing {len(chunks)} chunk(s) with Groq {s.transcription_model}")
        transcriber = GroqTranscriber(s, self.log, self.cancel_event)
        results = transcriber.transcribe_all(chunks, work / "chunks", p)
        transcript = merge_chunk_results(results, info.video_id, info.duration, s.transcription_model, self.log)
        transcript.fingerprint = fp
        if not transcript.transcript:
            raise RuntimeError("Transcription returned no speech. Is the video silent or music-only?")
        atomic_write_json(transcript_path, transcript.model_dump())
        atomic_write_text(work / "transcript.txt", transcript.to_txt(info.title))
        self.log.info("Transcript saved: %s", transcript_path)
        return transcript

    # ------------------------------------------------------------ rendering
    def _render_all(self, metas: List[ShortMetadata], info: VideoInfo, transcript: Transcript,
                    out_dir: Path) -> List[ShortMetadata]:
        if not metas:
            return metas
        p = self._stage("rendering", f"Rendering {len(metas)} Short(s)")
        total_seconds = sum(m.duration for m in metas)
        ensure_disk_space(out_dir, total_seconds * 3_000_000, self.s.min_free_disk_gb, "rendering")
        done = 0
        workers = max(1, min(self.s.render_workers, 4))
        lock = threading.Lock()
        results: Dict[int, ShortMetadata] = {}
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="render") as pool:
            futures = {pool.submit(self._produce_short, m, info, transcript, out_dir): m for m in metas}
            for fut in as_completed(futures):
                m = futures[fut]
                try:
                    results[m.rank] = fut.result()
                except JobCancelled:
                    for f in futures:
                        f.cancel()
                    raise
                except Exception as exc:
                    self.log.error("Short #%d failed: %s", m.rank, exc)
                    m.status, m.error = "failed", str(exc)[:500]
                    results[m.rank] = m
                with lock:
                    done += 1
                p(done / len(metas), f"Rendered {done}/{len(metas)} Shorts")
        return [results[m.rank] for m in metas]

    def _direct_segments(self, segments: List, transcript: Transcript, info: VideoInfo, llm: LLMClient,
                         hl_dir: Path) -> List[ClipCandidate]:
        """Clips at the exact times the brand asked for; the AI only directs hook text, title and edit."""
        clips: List[ClipCandidate] = []
        for i, seg in enumerate(segments, 1):
            start = max(0.0, seg.start)
            end = min(info.duration, seg.end)
            if end - start < 1.0:
                self.log.warning("Segment %d (%.0f–%.0fs) is outside the video (%.0fs) — skipped",
                                 i, seg.start, seg.end, info.duration)
                continue
            clips.append(ClipCandidate(id=f"g{i}", clip_start=start, hook_start=start, clip_end=end, score=80,
                                       category="other", reason=seg.note or "Segment specified by the campaign",
                                       title=transcript.text_between(start, end, max_chars=60) or f"Clip {i}"))
        if not clips:
            raise RuntimeError("None of the campaign's timings fall inside this video.")

        system, template = load_prompt(self.s.prompts_dir, "segment_direction.txt")
        blocks = [f"id: {c.id}\nclip_start: {c.clip_start:.1f}\nclip_end: {c.clip_end:.1f}\n"
                  f"brand note: {next((sg.note for k, sg in enumerate(segments, 1) if f'g{k}' == c.id), '')}\n"
                  f"transcript:\n{format_transcript_window(transcript, c.clip_start, c.clip_end)[:4000]}"
                  for c in clips]
        user = fill_template(template, TITLE=info.title, AUDIENCE=self.s.target_audience,
                             BRIEF=brief_prompt(self.s), SEGMENTS="\n\n".join(blocks))
        fp = fingerprint([llm.label, system, user])
        cache = hl_dir / "segment_direction.json"
        cached = read_json(cache)
        ids = {c.id for c in clips}

        def validate(data: Any) -> Dict[str, Dict]:
            items = data.get("clips") if isinstance(data, dict) else data
            out = {str(it.get("id")): it for it in (items or []) if isinstance(it, dict) and str(it.get("id")) in ids}
            if not out:
                raise ValueError("no usable entries")
            return out

        if cached and cached.get("fingerprint") == fp:
            direction = cached["result"]
        else:
            try:
                direction = llm.chat_json(system, user, validate=validate, attempts=3, what="Segment direction",
                                          cancel_event=self.cancel_event)
                atomic_write_json(cache, {"fingerprint": fp, "result": direction})
            except JobCancelled:
                raise
            except Exception as exc:
                self.log.warning("AI direction for campaign segments failed (%s); using defaults", exc)
                direction = {}

        out: List[ClipCandidate] = []
        for rank, c in enumerate(clips, 1):
            d = dict(direction.get(c.id, {}))
            for fixed in ("clip_start", "clip_end", "id"):
                d.pop(fixed, None)
            try:
                c = ClipCandidate.model_validate({**c.model_dump(), **{k: v for k, v in d.items() if v not in (None, "")}})
            except Exception:
                pass
            if c.hook_start is None or not (c.clip_start <= c.hook_start <= c.clip_end):
                c.hook_start = c.clip_start
            c.final_score = c.score
            c.selected, c.rank = True, rank
            self.log.info("  Campaign segment #%d %.1f–%.1fs (%.0fs) hook %.1fs: %s", rank, c.clip_start,
                          c.clip_end, c.duration, c.hook_start, c.title)
            out.append(c)
        return out

    def clip_settings(self, meta: ShortMetadata) -> Settings:
        """Settings with this Short's AI/you-resolved editing decisions applied."""
        layout = meta.layout or (self.s.layout if self.s.layout != "auto" else "blur_fit")
        punch = meta.punch_in if meta.punch_in is not None else (
            self.s.punch_in if self.s.punch_in_mode == "auto" else self.s.punch_in_mode == "on")
        zoom = meta.zoom if meta.zoom is not None else (
            self.s.zoom_effect if self.s.zoom_mode == "auto" else self.s.zoom_mode == "on")
        return self.s.model_copy(update={"layout": layout, "punch_in": punch, "zoom_effect": zoom})

    @staticmethod
    def _overlay_on(s: Settings) -> bool:
        return s.title_overlay and s.overlay_text != "off"

    def _produce_short(self, meta: ShortMetadata, info: VideoInfo, transcript: Transcript,
                       out_dir: Path) -> ShortMetadata:
        self._check_cancel()
        s = self.clip_settings(meta)
        render_fp = fingerprint([meta.hook_text, s.on_screen_text, s.logo_path, s.logo_position, s.logo_scale,
            meta.clip_start, meta.clip_end, meta.hook_start, meta.title, meta.caption_style, meta.keywords,
            s.output_width, s.output_height, s.fps, s.video_crf, s.layout, s.crop_x_position, s.foreground_zoom,
            s.captions_enabled, s.caption_font, s.caption_font_size, s.caption_position, s.title_overlay,
            s.zoom_effect, s.zoom_max, s.punch_in, s.punch_in_zoom, s.fade_transitions, s.normalize_audio,
            s.target_lufs, s.background_music, s.music_volume,
        ])
        final = out_dir / meta.filename
        sidecar = final.with_suffix(".json")
        previous = read_json(sidecar)
        if final.exists() and previous and previous.get("render_fingerprint") == render_fp \
                and previous.get("status") == "exported":
            self.log.info("Short #%d already rendered — skipping (%s)", meta.rank, final.name)
            prev = ShortMetadata.model_validate(previous)
            # keep the newest text metadata (it may have been regenerated with fresh trends)
            prev.description, prev.hashtags, prev.tags = meta.description, meta.hashtags, meta.tags
            prev.filename, prev.category, prev.score = meta.filename, meta.category, meta.score
            atomic_write_json(sidecar, prev.model_dump())
            return prev

        workdir = s.clips_dir / info.video_id / f"{meta.rank:02d}_{fingerprint([meta.clip_start, meta.clip_end])[:8]}"
        workdir.mkdir(parents=True, exist_ok=True)
        raw = extract_clip(Path(info.source_path), meta.clip_start, meta.clip_end, workdir / "raw.mp4",
                           s.min_free_disk_gb, self.log)
        duration = meta.clip_end - meta.clip_start
        hook_rel = meta.hook_start - meta.clip_start

        # words relative to the clip
        words = [w.model_copy(update={"start": w.start - meta.clip_start, "end": w.end - meta.clip_start})
                 for w in transcript.words_between(meta.clip_start, meta.clip_end)]
        keyset = {k.lower() for kw in meta.keywords for k in kw.split()}
        keyword_times = [w.start for w in words if w.word.lower().strip(".,!?") in keyset]

        if words:
            atomic_write_text(final.with_suffix(".srt"), build_srt(words, duration))

        editor = VerticalEditor(s, self.log)
        plan = editor.plan(raw, hook_rel, duration, keyword_times)
        for note in plan.notes:
            self.log.info("Short #%d: %s", meta.rank, note)

        ass_path: Optional[Path] = None
        spans: List[Tuple[float, float]] = []
        if s.captions_enabled or self._overlay_on(s) or s.on_screen_text:
            gen = CaptionGenerator(meta.caption_style, s.caption_font, s.caption_font_size, plan.caption_y)
            ass_text = gen.build(
                words if s.captions_enabled else [],
                duration,
                keywords=meta.keywords,
                title=(meta.hook_text or meta.title) if self._overlay_on(s) else "",
                title_y=plan.title_y,
                title_duration=None if plan.title_full_duration else 3.0,
                footer_text=s.on_screen_text,
            )
            ass_path = workdir / "captions.ass"
            ass_path.write_text(ass_text, encoding="utf-8")
            spans = caption_spans(ass_text)

        qc = QualityChecker(s.output_width, s.output_height, s.min_clip_duration, s.hard_max_duration, self.log)
        tmp = final.with_name(final.stem + ".rendering.mp4")
        last_error = ""
        for mode in ("full", "safe", "no_captions"):
            if mode == "no_captions" and ass_path is None:
                break
            self._check_cancel()
            try:
                self.log.info("Rendering Short #%d (%s, %.0fs, mode=%s)", meta.rank, meta.caption_style, duration, mode)
                editor.render(raw, tmp, plan, ass_path, mode=mode)
            except JobCancelled:
                raise
            except Exception as exc:
                last_error = f"render ({mode}) failed: {exc}"
                self.log.warning("Short #%d: %s", meta.rank, last_error[:600])
                remove_path(tmp)
                continue
            result = qc.check(tmp, duration, spans if mode != "no_captions" else None,
                              captions_expected=s.captions_enabled and mode != "no_captions")
            if result.passed:
                os.replace(tmp, final)
                meta.status, meta.render_mode, meta.error = "exported", mode, ""
                meta.output_path = str(final)
                meta.qc = result.to_dict()
                meta.render_fingerprint = render_fp
                if mode == "no_captions":
                    meta.qc.setdefault("warnings", []).append("exported WITHOUT captions (caption rendering failed)")
                    self.log.error("Short #%d exported without captions — check fonts/libass", meta.rank)
                for w in result.warnings:
                    self.log.warning("Short #%d QC warning: %s", meta.rank, w)
                thumb = editor.thumbnail(final, min(hook_rel + 0.6, duration - 0.1), final.with_suffix(".jpg"))
                meta.thumbnail_path = str(thumb) if thumb else ""
                atomic_write_json(sidecar, meta.model_dump())
                self.log.info("✔ Short #%d exported: %s (%.1f MB)", meta.rank, final.name, result.info.get("size_mb", 0))
                if not s.keep_intermediate:
                    remove_path(workdir)
                return meta
            last_error = "QC failed: " + "; ".join(result.errors)
            self.log.warning("Short #%d (%s): %s", meta.rank, mode, last_error)
            remove_path(tmp)

        meta.status, meta.error = "failed", last_error[:1000]
        atomic_write_json(sidecar, meta.model_dump())
        self.log.error("✘ Short #%d was NOT exported: %s", meta.rank, last_error[:300])
        return meta

    # ------------------------------------------------------------ upload
    def _auto_upload(self) -> None:
        """Upload exported Shorts to YouTube. Upload problems never fail the job (files are kept)."""
        from modules.youtube_uploader import YouTubeUploader

        self._stage("upload", "Uploading Shorts to YouTube")
        try:
            results = YouTubeUploader(self.s, self.db, self.log, self.cancel_event).upload_job(
                self.job_id, progress=lambda f, m: self._progress("upload", f, m))
            ok = sum(1 for r in results if r.get("upload_status") in ("uploaded", "scheduled"))
            self.log.info("YouTube: %d/%d Short(s) uploaded", ok, len(results))
        except JobCancelled:
            raise
        except Exception as exc:
            self.log.error("YouTube upload failed: %s — the Shorts are saved; upload later with "
                           "'python main.py upload %s' or the YouTube tab", exc, self.job_id)

    # ------------------------------------------------------------ cleanup
    def _cleanup(self, info: VideoInfo, work: Path) -> None:
        if self.s.keep_intermediate:
            return
        remove_path(work / "audio_16k.wav")
        remove_path(work / "audio_chunks")
        clip_dir = self.s.clips_dir / info.video_id
        if clip_dir.exists() and not any(clip_dir.iterdir()):
            remove_path(clip_dir)
        self.log.info("Temporary audio files cleaned up (transcripts and analysis are kept)")
