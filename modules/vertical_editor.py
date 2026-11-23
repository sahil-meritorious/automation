"""Turn a horizontal (or any) clip into a polished 9:16 Short with FFmpeg.

Layouts
  blur_fit     – full frame (HUD preserved) over a blurred, darkened fill. Best for gameplay.
  center_crop  – fill the frame by cropping the centre.
  offset_crop  – fill the frame, crop position set by CROP_X_POSITION (0=left … 1=right).
  face_track   – subject-aware crop: detects faces (OpenCV, optional) and centres the crop
                 on them; falls back to blur_fit when no stable face is found.

Effects (all optional): slow zoom, punch-in at the hook, fades, loudness normalisation,
background music (only use audio you have the rights to).
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from config import Settings
from core.utils import escape_filter_path, run_ffmpeg
from modules.video_metadata import probe_video

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus"}


@dataclass
class RenderPlan:
    layout: str
    crop_x: float = 0.5  # 0..1 position of the crop window (crop layouts)
    fg_height: int = 0  # height of the visible foreground band (blur_fit)
    caption_y: float = 0  # caption centre on the 1080x1920 caption canvas
    title_y: float = 0
    title_full_duration: bool = True
    punch_times: List[float] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def find_music(path_setting: str) -> Optional[Path]:
    if not path_setting:
        return None
    p = Path(path_setting).expanduser()
    if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
        return p
    if p.is_dir():
        files = [f for f in p.iterdir() if f.suffix.lower() in AUDIO_EXTS]
        return random.choice(files) if files else None
    return None


class VerticalEditor:
    def __init__(self, settings: Settings, logger: Optional[logging.Logger] = None) -> None:
        self.s = settings
        self.log = logger or logging.getLogger("shorts.editor")
        self.W, self.H = settings.output_width, settings.output_height

    # ------------------------------------------------------------ planning
    def _face_crop_position(self, clip: Path, src_w: int, src_h: int) -> Optional[float]:
        try:
            import cv2  # optional dependency
        except ImportError:
            self.log.info("OpenCV not installed — face_track falls back to blur_fit")
            return None
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(str(cascade_path))
        cap = cv2.VideoCapture(str(clip))
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(fps * 0.5))
        centers, samples = [], 0
        for frame_idx in range(0, max(total, 1), step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break
            samples += 1
            small = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6, minSize=(36, 36))
            if len(faces):
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                centers.append((x + w / 2) / small.shape[1])
            if samples >= 160:
                break
        cap.release()
        if samples == 0 or len(centers) < max(3, samples * 0.3):
            return None
        centers.sort()
        cx = centers[len(centers) // 2]  # median = stable framing, no jitter
        crop_frac = (src_h * self.W / self.H) / src_w  # crop width as fraction of the source width
        if crop_frac >= 1:
            return 0.5
        return min(1.0, max(0.0, (cx - crop_frac / 2) / (1 - crop_frac)))

    def plan(self, clip: Path, hook_rel: float, duration: float, keyword_times: List[float]) -> RenderPlan:
        meta = probe_video(clip)
        src_w, src_h = meta["width"] or 1920, meta["height"] or 1080
        layout = self.s.layout if self.s.layout != "auto" else "blur_fit"
        plan = RenderPlan(layout=layout, crop_x=self.s.crop_x_position)

        if layout == "center_crop":
            plan.crop_x = 0.5
        elif layout == "face_track":
            pos = self._face_crop_position(clip, src_w, src_h)
            if pos is None:
                plan.layout = "blur_fit"
                plan.notes.append("face_track: no stable face found, used blur_fit")
            else:
                plan.layout, plan.crop_x = "offset_crop", pos
                plan.notes.append(f"face_track: crop position {pos:.2f}")

        if src_w / src_h <= self.W / self.H + 0.01 and plan.layout != "blur_fit":
            plan.crop_x = 0.5  # source already vertical: nothing to crop horizontally

        # Caption / title placement on the 1080x1920 caption canvas
        canvas_h = 1920
        if plan.layout == "blur_fit":
            scale = min(self.W / src_w, self.H / src_h) * max(1.0, self.s.foreground_zoom)
            fg_h = min(self.H, int(src_h * scale))
            plan.fg_height = fg_h
            fg_bottom = (self.H + fg_h) / 2 / self.H * canvas_h
            fg_top = (self.H - fg_h) / 2 / self.H * canvas_h
            plan.caption_y = min(canvas_h * 0.78, fg_bottom + 150) if fg_bottom < canvas_h * 0.75 else canvas_h * 0.72
            plan.title_y = max(canvas_h * 0.1, fg_top * 0.55) if fg_top > canvas_h * 0.15 else canvas_h * 0.14
            plan.title_full_duration = fg_top > canvas_h * 0.15
        else:
            plan.caption_y = canvas_h * 0.70
            plan.title_y = canvas_h * 0.14
            plan.title_full_duration = False
        if self.s.caption_position:
            plan.caption_y = canvas_h * min(0.9, max(0.1, self.s.caption_position))

        if self.s.punch_in:
            times = []
            if 0.4 < hook_rel < duration - 1.5:
                times.append(hook_rel)
            for t in keyword_times:
                if len(times) >= 3:
                    break
                if 1.0 < t < duration - 1.5 and all(abs(t - x) > 4 for x in times):
                    times.append(t)
            plan.punch_times = sorted(times)
        logo = self.logo_file()
        if logo is not None and self.s.logo_position.startswith("top"):
            try:
                lm = probe_video(logo)
                ratio = (lm["height"] or 1) / (lm["width"] or 1)
            except Exception:
                ratio = 0.5
            canvas_w = 1080  # caption canvas is 1080x1920
            logo_bottom = 1920 * 0.075 + canvas_w * self.s.logo_scale * ratio
            if plan.title_y - 60 < logo_bottom:
                plan.title_y = logo_bottom + 70
                plan.notes.append("title moved below the logo")
        return plan

    # ------------------------------------------------------------ filters
    def logo_file(self) -> Optional[Path]:
        if not self.s.logo_path:
            return None
        p = Path(self.s.logo_path).expanduser()
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
            return p
        self.log.warning("Logo file not found or unsupported: %s", self.s.logo_path)
        return None

    def _logo_xy(self) -> tuple:
        mx, top, bottom = int(self.W * 0.045), int(self.H * 0.075), int(self.H * 0.20)  # clear of Shorts UI
        return {
            "top_left": (f"{mx}", f"{top}"),
            "top_center": ("(W-w)/2", f"{top}"),
            "bottom_left": (f"{mx}", f"H-h-{bottom}"),
            "bottom_right": (f"W-w-{mx}", f"H-h-{bottom}"),
        }.get(self.s.logo_position, (f"W-w-{mx}", f"{top}"))

    def _video_filters(self, plan: RenderPlan, duration: float, ass_name: Optional[str], effects: bool,
                       logo_idx: Optional[int] = None) -> str:
        W, H, F = self.W, self.H, self.s.fps
        chains = []
        if plan.layout == "blur_fit":
            z = max(1.0, self.s.foreground_zoom)
            chains.append("[0:v]split=2[bgsrc][fgsrc]")
            chains.append(
                f"[bgsrc]scale={W // 4}:{H // 4}:force_original_aspect_ratio=increase,crop={W // 4}:{H // 4},"
                f"boxblur=12:3,scale={W}:{H},eq=brightness=-0.10:saturation=1.15,setsar=1[bg]"
            )
            chains.append(
                f"[fgsrc]scale=w='trunc(min({W}/iw\\,{H}/ih)*iw*{z}/2)*2':h='trunc(min({W}/iw\\,{H}/ih)*ih*{z}/2)*2',"
                f"crop='min(iw\\,{W})':'min(ih\\,{H})',setsar=1[fg]"
            )
            chains.append("[bg][fg]overlay=(W-w)/2:(H-h)/2:format=auto[base]")
        else:
            chains.append(
                f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
                f"crop={W}:{H}:x='(iw-ow)*{plan.crop_x:.3f}':y='(ih-oh)/2',setsar=1[base]"
            )
        cur = "base"
        chains.append(f"[{cur}]fps={F}[v_fps]")
        cur = "v_fps"

        if effects and self.s.zoom_effect and duration > 2:
            rate = (max(1.0, self.s.zoom_max) - 1.0) / max(1.0, duration * F)
            chains.append(
                f"[{cur}]zoompan=z='min(1+{rate:.7f}*on\\,{self.s.zoom_max})':x='iw/2-(iw/zoom/2)':"
                f"y='ih/2-(ih/zoom/2)':d=1:s={W}x{H}:fps={F}[v_zoom]"
            )
            cur = "v_zoom"

        if effects and plan.punch_times:
            z = max(1.01, self.s.punch_in_zoom)
            zw, zh = int(W * z) // 2 * 2, int(H * z) // 2 * 2
            enable = "+".join(f"between(t\\,{t:.2f}\\,{t + 1.1:.2f})" for t in plan.punch_times)
            chains.append(f"[{cur}]split=2[pa][pb]")
            chains.append(f"[pb]scale={zw}:{zh},crop={W}:{H}[pz]")
            chains.append(f"[pa][pz]overlay=0:0:enable='{enable}'[v_punch]")
            cur = "v_punch"

        if logo_idx is not None:
            lw = max(32, int(self.W * min(0.5, max(0.05, self.s.logo_scale))) // 2 * 2)
            alpha = min(1.0, max(0.1, self.s.logo_opacity))
            x, y = self._logo_xy()
            chains.append(f"[{logo_idx}:v]scale={lw}:-2,format=rgba,colorchannelmixer=aa={alpha:.2f}[logo]")
            chains.append(f"[{cur}][logo]overlay={x}:{y}:format=auto:shortest=1[v_logo]")
            cur = "v_logo"

        if ass_name:
            fonts = ""
            if self.s.fonts_dir.exists() and any(self.s.fonts_dir.iterdir()):
                fonts = f":fontsdir={escape_filter_path(self.s.fonts_dir)}"
            chains.append(f"[{cur}]subtitles=filename={ass_name}{fonts}[v_sub]")
            cur = "v_sub"

        post = []
        if effects and self.s.fade_transitions and duration > 3:
            post.append("fade=t=in:st=0:d=0.25")
            post.append(f"fade=t=out:st={max(0.0, duration - 0.35):.2f}:d=0.35")
        post.append("format=yuv420p")
        chains.append(f"[{cur}]{','.join(post)}[vout]")
        return ";".join(chains)

    def _audio_filters(self, has_audio: bool, music: Optional[Path], duration: float, effects: bool) -> str:
        chains = []
        voice = "[0:a]" if has_audio else "[1:a]"
        chains.append(f"{voice}aresample=48000,aformat=channel_layouts=stereo[a_voice]")
        cur = "a_voice"
        if effects and music is not None:
            m_idx = 1 if has_audio else 2
            fade_out = max(0.0, duration - 1.5)
            chains.append(
                f"[{m_idx}:a]aresample=48000,aformat=channel_layouts=stereo,atrim=0:{duration:.2f},"
                f"volume={self.s.music_volume:.3f},afade=t=in:st=0:d=1,afade=t=out:st={fade_out:.2f}:d=1.5[a_music]"
            )
            chains.append(f"[{cur}][a_music]amix=inputs=2:duration=first:dropout_transition=0,volume=2[a_mix]")
            cur = "a_mix"
        post = []
        if self.s.normalize_audio:
            post.append(f"loudnorm=I={self.s.target_lufs}:TP=-1.5:LRA=11")
            post.append("aresample=48000")
        if effects and self.s.fade_transitions and duration > 3:
            post.append("afade=t=in:st=0:d=0.15")
            post.append(f"afade=t=out:st={max(0.0, duration - 0.35):.2f}:d=0.35")
        post.append("asetpts=PTS-STARTPTS")
        chains.append(f"[{cur}]{','.join(post)}[aout]")
        return ";".join(chains)

    # ------------------------------------------------------------ render
    def render(
        self,
        clip: Path,
        out_path: Path,
        plan: RenderPlan,
        ass_path: Optional[Path],
        mode: str = "full",
    ) -> Path:
        """mode: 'full' (all effects), 'safe' (no effects/music), 'no_captions' (safe + no subtitles)."""
        meta = probe_video(clip)
        duration = meta["duration"]
        has_audio = meta["has_audio"]
        effects = mode == "full"
        use_ass = ass_path is not None and mode != "no_captions"
        music = find_music(self.s.background_music) if effects else None

        inputs = ["-i", clip.name]
        if not has_audio:
            inputs += ["-f", "lavfi", "-t", f"{duration:.3f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
        if music is not None:
            inputs += ["-stream_loop", "-1", "-i", str(music.resolve())]
        logo = self.logo_file()
        logo_idx = None
        if logo is not None:  # brand requirement: applied in every render mode
            logo_idx = 1 + (0 if has_audio else 1) + (1 if music is not None else 0)
            inputs += ["-loop", "1", "-t", f"{duration:.3f}", "-i", str(logo.resolve())]

        graph = self._video_filters(plan, duration, ass_path.name if use_ass else None, effects, logo_idx)
        graph += ";" + self._audio_filters(has_audio, music, duration, effects)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        args = [
            *inputs,
            "-filter_complex", graph,
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", self.s.video_preset, "-crf", str(self.s.video_crf),
            "-profile:v", "high", "-pix_fmt", "yuv420p", "-r", str(self.s.fps),
            "-c:a", "aac", "-b:a", self.s.audio_bitrate, "-ar", "48000",
            "-t", f"{duration:.3f}",
            "-movflags", "+faststart",
            str(out_path.resolve()),
        ]
        # Run inside the clip folder so the subtitle filename needs no path escaping (Windows-safe)
        run_ffmpeg(args, cwd=clip.parent, timeout=max(600, duration * 30), logger=self.log)
        return out_path

    def thumbnail(self, video: Path, at: float, out: Path) -> Optional[Path]:
        try:
            run_ffmpeg(["-ss", f"{max(0.0, at):.2f}", "-i", str(video), "-frames:v", "1", "-q:v", "3", str(out)],
                       timeout=60, logger=self.log)
            return out
        except Exception as exc:  # thumbnails are a nice-to-have
            self.log.warning("Thumbnail failed: %s", exc)
            return None
