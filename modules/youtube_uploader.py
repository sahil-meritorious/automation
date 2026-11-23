"""Upload finished Shorts to YouTube with the Data API v3 (OAuth 2.0).

* One-time browser consent (``python main.py youtube-auth`` or the UI button);
  the refresh token is stored in database/youtube_token.json (chmod 600, git-ignored).
* Resumable, chunked uploads with retry/back-off on network and 5xx errors.
* Title, description (+ footer + hashtags), tags, category, language, made-for-kids flag.
* Optional SRT subtitle track (captions.insert) and custom thumbnail.
* Optional scheduling: one Short every N hours via ``publishAt``.
* Idempotent: a Short that already has a YouTube id is never uploaded twice (unless forced).
* Quota errors stop the batch cleanly; remaining Shorts stay "pending" for tomorrow.

Only upload content you own or are licensed to republish.
"""
from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from config import Settings
from core.utils import JobCancelled, atomic_write_json, interruptible_sleep, read_json
from database.database import Database, get_database
from modules.metadata_generator import build_description, clean_tags

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",  # captions.insert + channel info
]
RETRIABLE_STATUS = {500, 502, 503, 504}
QUOTA_REASONS = {"quotaExceeded", "uploadLimitExceeded", "dailyLimitExceeded", "rateLimitExceeded",
                 "userRateLimitExceeded"}
ProgressFn = Callable[[float, str], None]


class YouTubeError(RuntimeError):
    pass


class NotAuthorizedError(YouTubeError):
    pass


class QuotaExceededError(YouTubeError):
    pass


def _require_libs() -> None:
    try:
        import google_auth_oauthlib  # noqa: F401
        import googleapiclient  # noqa: F401
    except ImportError as exc:
        raise YouTubeError("Google API libraries missing. Run: pip install -r requirements.txt") from exc


def pacific_day() -> str:
    """YouTube quota resets at midnight Pacific time."""
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
    except Exception:
        return (datetime.now(timezone.utc) - timedelta(hours=8)).strftime("%Y-%m-%d")


# ------------------------------------------------------------------ OAuth
def client_config(s: Settings) -> Dict[str, Any]:
    if s.youtube_client_secrets_file:
        path = Path(s.youtube_client_secrets_file).expanduser()
        if not path.exists():
            raise NotAuthorizedError(f"YOUTUBE_CLIENT_SECRETS_FILE not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if "installed" not in data and "web" not in data:
            raise NotAuthorizedError("Client secrets file must be an OAuth client JSON (Desktop app or Web)")
        return data
    if not (s.youtube_client_id and s.youtube_client_secret):
        raise NotAuthorizedError(
            "Uploading needs an OAuth client: set YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET "
            "(or YOUTUBE_CLIENT_SECRETS_FILE). An API key alone cannot upload videos.")
    return {"installed": {
        "client_id": s.youtube_client_id.strip(),
        "client_secret": s.youtube_client_secret.strip(),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["http://localhost"],
    }}


def _save_token(path: Path, creds: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(creds.to_json(), encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    tmp.replace(path)


OAUTH_ERRORS = {
    "access_denied": (
        "Google returned 'access_denied'. Usually this means the Google account is not a Test user. Open "
        "Google Cloud → OAuth consent screen → Audience, add your channel's Google account under Test users "
        "(or publish the app), then try again. It also happens if you pressed Cancel or did not tick the "
        "YouTube permission checkboxes."),
    "redirect_uri_mismatch": (
        "redirect_uri_mismatch: add exactly {uri} to the OAuth client's Authorized redirect URIs "
        "(including the trailing slash), or set YOUTUBE_REDIRECT_URI to the URI you registered."),
}


def _finish(s: Settings, creds: Any) -> Dict[str, str]:
    if not creds.refresh_token:
        raise NotAuthorizedError("Google did not return a refresh token. Remove the app at "
                                 "https://myaccount.google.com/permissions and connect again.")
    _save_token(s.youtube_token_path, creds)
    return channel_info(s)


def authorize(s: Settings, open_browser: bool = True) -> Dict[str, str]:
    """CLI consent flow: temporary local server on YOUTUBE_OAUTH_PORT. Returns channel info."""
    _require_libs()
    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_config(client_config(s), SCOPES)
    try:
        creds = flow.run_local_server(
            host="localhost", port=s.youtube_oauth_port, open_browser=open_browser,
            access_type="offline", prompt="consent", timeout_seconds=300,
            authorization_prompt_message="Open this URL in your browser to connect your YouTube channel:\n{url}\n",
            success_message="YouTube connected. You can close this tab and return to the Shorts generator.",
        )
    except OSError as exc:
        if getattr(exc, "errno", None) in (98, 48, 10048):
            raise NotAuthorizedError(
                f"Port {s.youtube_oauth_port} is in use (an earlier sign-in is still waiting). Close that "
                f"terminal/app or set YOUTUBE_OAUTH_PORT to another port.") from exc
        raise
    except Exception as exc:
        if "access_denied" in str(exc):
            raise NotAuthorizedError(OAUTH_ERRORS["access_denied"]) from exc
        raise
    if creds is None:
        raise NotAuthorizedError("Sign-in timed out after 5 minutes — run youtube-auth again.")
    return _finish(s, creds)


# ---- Streamlit (web redirect) flow: Google redirects back to the app itself, no extra server/port.
PENDING_TTL = 900


def _pending_path(s: Settings) -> Path:
    return s.data_dir / "database" / "youtube_oauth_pending.json"


def start_web_auth(s: Settings, redirect_uri: Optional[str] = None) -> str:
    """Return the Google sign-in URL. The PKCE verifier is stored server-side, keyed by `state`,
    because the browser leaves the app (new Streamlit session) during sign-in."""
    _require_libs()
    from google_auth_oauthlib.flow import Flow

    redirect_uri = redirect_uri or s.youtube_redirect_uri
    flow = Flow.from_client_config(client_config(s), SCOPES, redirect_uri=redirect_uri)
    url, state = flow.authorization_url(access_type="offline", prompt="consent", include_granted_scopes="true")
    path = _pending_path(s)
    now = time.time()
    pending = {k: v for k, v in (read_json(path) or {}).items() if now - v.get("created", 0) < PENDING_TTL}
    pending[state] = {"code_verifier": flow.code_verifier, "redirect_uri": redirect_uri, "created": now}
    atomic_write_json(path, pending)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return url


def finish_web_auth(s: Settings, code: str, state: str) -> Dict[str, str]:
    """Exchange the ?code=… Google sent back for tokens. Returns channel info."""
    _require_libs()
    from google_auth_oauthlib.flow import Flow

    path = _pending_path(s)
    pending = read_json(path) or {}
    entry = pending.pop(state, None)
    atomic_write_json(path, pending)
    if not entry or time.time() - entry.get("created", 0) > PENDING_TTL:
        raise NotAuthorizedError("This sign-in link expired or was already used. Press 'Connect YouTube' again.")
    os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")  # Google may add previously granted scopes
    flow = Flow.from_client_config(client_config(s), SCOPES, redirect_uri=entry["redirect_uri"], state=state)
    flow.code_verifier = entry["code_verifier"]
    try:
        flow.fetch_token(code=code)
    except Exception as exc:
        msg = str(exc)
        if "redirect_uri_mismatch" in msg:
            raise NotAuthorizedError(OAUTH_ERRORS["redirect_uri_mismatch"].format(uri=entry["redirect_uri"])) from exc
        if "invalid_client" in msg:
            raise NotAuthorizedError("invalid_client: the client ID / secret pair is wrong. Copy both again "
                                     "from the same OAuth client in Google Cloud → Credentials.") from exc
        raise NotAuthorizedError(f"Token exchange failed: {msg}") from exc
    return _finish(s, flow.credentials)


def oauth_error_message(error: str, s: Settings) -> str:
    return OAUTH_ERRORS.get(error, f"Google sign-in failed: {error}").format(uri=s.youtube_redirect_uri)


def disconnect(s: Settings) -> None:
    s.youtube_token_path.unlink(missing_ok=True)


def load_credentials(s: Settings) -> Any:
    _require_libs()
    from google.auth.exceptions import RefreshError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    path = s.youtube_token_path
    if not path.exists():
        raise NotAuthorizedError("YouTube is not connected. Run 'python main.py youtube-auth' "
                                 "or press 'Connect YouTube' in the app.")
    creds = Credentials.from_authorized_user_file(str(path), SCOPES)
    if not creds.valid:
        if not creds.refresh_token:
            raise NotAuthorizedError("Stored YouTube token has no refresh token — connect again.")
        try:
            creds.refresh(Request())
        except RefreshError as exc:
            raise NotAuthorizedError(
                f"YouTube authorisation expired or was revoked ({exc}). Connect again. Tip: if your OAuth "
                f"consent screen is in 'Testing' mode, Google expires tokens after 7 days — publish the app "
                f"(In production) to keep it connected.") from exc
        _save_token(path, creds)
    return creds


def youtube_service(s: Settings) -> Any:
    from googleapiclient.discovery import build

    return build("youtube", "v3", credentials=load_credentials(s), cache_discovery=False)


def channel_info(s: Settings) -> Dict[str, str]:
    yt = youtube_service(s)
    items = yt.channels().list(part="snippet,statistics", mine=True).execute().get("items", [])
    if not items:
        raise YouTubeError("This Google account has no YouTube channel. Create one at youtube.com first.")
    ch = items[0]
    return {"id": ch["id"], "title": ch["snippet"]["title"],
            "subscribers": ch.get("statistics", {}).get("subscriberCount", "?"),
            "url": f"https://www.youtube.com/channel/{ch['id']}"}


def auth_status(s: Settings) -> Dict[str, Any]:
    """{'connected': bool, 'message': str, 'channel': {...}} — never raises."""
    try:
        ch = channel_info(s)
        return {"connected": True, "message": f"Connected to {ch['title']}", "channel": ch}
    except Exception as exc:
        return {"connected": False, "message": str(exc), "channel": None}


def _http_error_reason(exc: Any) -> str:
    try:
        data = json.loads(exc.content.decode("utf-8"))
        errs = data.get("error", {}).get("errors") or [{}]
        return errs[0].get("reason", "") or data.get("error", {}).get("status", "")
    except Exception:
        return ""


def _http_error_message(exc: Any) -> str:
    try:
        return json.loads(exc.content.decode("utf-8"))["error"]["message"]
    except Exception:
        return str(exc)


# ------------------------------------------------------------------ uploader
class YouTubeUploader:
    def __init__(
        self,
        settings: Settings,
        db: Optional[Database] = None,
        logger: Optional[logging.Logger] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        self.s = settings
        self.db = db or get_database(settings.db_path)
        self.log = logger or logging.getLogger("shorts.youtube")
        self.cancel_event = cancel_event
        self._yt = None

    @property
    def yt(self) -> Any:
        if self._yt is None:
            self._yt = youtube_service(self.s)
        return self._yt

    def _check_cancel(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise JobCancelled("Upload cancelled")

    def _spend(self, units: int) -> None:
        used = self.db.add_quota(pacific_day(), units)
        if used > self.s.youtube_daily_quota:
            self.log.warning("Estimated YouTube quota used today: %d / %d units", used, self.s.youtube_daily_quota)

    def _execute(self, request: Any, what: str) -> Any:
        from googleapiclient.errors import HttpError

        for attempt in range(1, 6):
            try:
                return request.execute()
            except HttpError as exc:
                reason = _http_error_reason(exc)
                if reason in QUOTA_REASONS:
                    raise QuotaExceededError(f"{what}: {_http_error_message(exc)} ({reason})") from exc
                if exc.resp.status in RETRIABLE_STATUS and attempt < 5:
                    interruptible_sleep(2 ** attempt + random.random(), self.cancel_event)
                    continue
                raise YouTubeError(f"{what}: {_http_error_message(exc)} ({reason or exc.resp.status})") from exc
            except (OSError, TimeoutError) as exc:
                if attempt >= 5:
                    raise YouTubeError(f"{what}: network error {exc}") from exc
                interruptible_sleep(2 ** attempt + random.random(), self.cancel_event)
        raise YouTubeError(f"{what}: failed")

    # -------------------------------------------------------------- one video
    def build_body(self, meta: Dict[str, Any], privacy: str, publish_at: Optional[datetime]) -> Dict[str, Any]:
        class _M:  # adapter so build_description works with dicts
            description = meta.get("description", "")
            hashtags = meta.get("hashtags", [])

        title = str(meta.get("title") or "New Short").replace("<", "").replace(">", "").strip()[:100]
        lang = self.s.youtube_language or None
        snippet: Dict[str, Any] = {
            "title": title,
            "description": build_description(_M, self.s.youtube_description_footer).replace("<", "").replace(">", ""),
            "tags": clean_tags(meta.get("tags") or []),
            "categoryId": str(self.s.youtube_category_id or "20"),
        }
        if lang:
            snippet["defaultLanguage"] = lang
            snippet["defaultAudioLanguage"] = lang
        status: Dict[str, Any] = {
            "privacyStatus": "private" if publish_at else privacy,
            "selfDeclaredMadeForKids": bool(self.s.youtube_made_for_kids),
            "embeddable": True,
            "license": "youtube",
        }
        if publish_at:
            status["publishAt"] = publish_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        return {"snippet": snippet, "status": status}

    def upload_video(self, video_path: Path, body: Dict[str, Any], progress: Optional[ProgressFn] = None) -> str:
        from googleapiclient.errors import HttpError
        from googleapiclient.http import MediaFileUpload

        media = MediaFileUpload(str(video_path), mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True)
        request = self.yt.videos().insert(part="snippet,status", body=body, media_body=media,
                                          notifySubscribers=True)
        response = None
        retries = 0
        while response is None:
            self._check_cancel()
            try:
                status, response = request.next_chunk()
                if status and progress:
                    progress(status.progress(), f"Uploading {video_path.name} {status.progress() * 100:.0f}%")
            except HttpError as exc:
                reason = _http_error_reason(exc)
                if reason in QUOTA_REASONS:
                    raise QuotaExceededError(f"{_http_error_message(exc)} ({reason})") from exc
                if exc.resp.status not in RETRIABLE_STATUS:
                    raise YouTubeError(f"Upload rejected: {_http_error_message(exc)} ({reason or exc.resp.status})") from exc
                retries += 1
                err = exc
            except (OSError, TimeoutError, ConnectionError) as exc:
                retries += 1
                err = exc
            else:
                continue
            if retries > 8:
                raise YouTubeError(f"Upload failed after {retries} retries: {err}")
            delay = min(64, 2 ** retries) * (0.8 + random.random() * 0.4)
            self.log.warning("Upload hiccup (%s) — retry %d in %.0fs", err, retries, delay)
            interruptible_sleep(delay, self.cancel_event)
        if "id" not in response:
            raise YouTubeError(f"Unexpected upload response: {response}")
        self._spend(self.s.youtube_upload_cost)
        return response["id"]

    def upload_captions(self, video_id: str, srt: Path) -> None:
        from googleapiclient.http import MediaFileUpload

        body = {"snippet": {"videoId": video_id, "language": self.s.youtube_language or "en",
                            "name": "", "isDraft": False}}
        req = self.yt.captions().insert(part="snippet", body=body, sync=False,
                                        media_body=MediaFileUpload(str(srt), mimetype="application/octet-stream"))
        self._execute(req, "Caption upload")
        self._spend(self.s.youtube_caption_cost)

    def set_thumbnail(self, video_id: str, image: Path) -> None:
        from googleapiclient.http import MediaFileUpload

        req = self.yt.thumbnails().set(videoId=video_id, media_body=MediaFileUpload(str(image), mimetype="image/jpeg"))
        self._execute(req, "Thumbnail")
        self._spend(50)

    # -------------------------------------------------------------- batch
    def _schedule(self, count: int, start: Optional[datetime], interval_hours: float) -> List[Optional[datetime]]:
        if interval_hours <= 0 and start is None:
            return [None] * count
        now = datetime.now(timezone.utc)
        if start is None:
            last = self.db.last_scheduled_at()
            start = now + timedelta(minutes=20)
            if last:
                try:
                    nxt = datetime.fromisoformat(last) + timedelta(hours=interval_hours)
                    start = max(start, nxt)
                except ValueError:
                    pass
        if start.tzinfo is None:
            start = start.astimezone()  # local time → aware
        start = max(start, now + timedelta(minutes=15))  # publishAt must be in the future
        step = timedelta(hours=max(interval_hours, 0))
        return [start + step * i for i in range(count)]

    def upload_job(
        self,
        job_id: str,
        ranks: Optional[List[int]] = None,
        privacy: Optional[str] = None,
        schedule_start: Optional[datetime] = None,
        interval_hours: Optional[float] = None,
        force: bool = False,
        progress: Optional[ProgressFn] = None,
    ) -> List[Dict[str, Any]]:
        job = self.db.get_job(job_id)
        if not job or not job.get("output_dir"):
            raise YouTubeError(f"Job {job_id} has no output folder")
        out_dir = Path(job["output_dir"])
        metas = read_json(out_dir / "metadata.json") or []
        todo = [m for m in metas if m.get("status") == "exported" and Path(m.get("output_path") or "").exists()
                and (not ranks or m["rank"] in ranks)]
        if not todo:
            raise YouTubeError("No exported Shorts to upload in this job")
        privacy = (privacy or self.s.youtube_privacy).lower()
        interval = self.s.youtube_schedule_interval_hours if interval_hours is None else interval_hours

        pending = []
        results: List[Dict[str, Any]] = []
        for m in sorted(todo, key=lambda x: x["rank"]):
            row = self.db.get_short(job_id, m["rank"]) or {}
            if row.get("youtube_video_id") and not force:
                self.log.info("Short #%d already on YouTube: %s — skipping", m["rank"], row["youtube_video_id"])
                results.append({**self._row_result(m, row), "skipped": True})
            else:
                pending.append(m)

        wants_schedule = privacy == "public" and (interval > 0 or schedule_start is not None)
        if (interval > 0 or schedule_start) and privacy != "public":
            self.log.info("Scheduling only applies to public uploads — uploading as %s without a publish time", privacy)
        times = self._schedule(len(pending), schedule_start, interval) if wants_schedule else [None] * len(pending)

        self.yt  # authenticate once up front (clear error before any work)
        for i, (m, when) in enumerate(zip(pending, times)):
            self._check_cancel()
            rank = m["rank"]
            path = Path(m["output_path"])

            def sub(f: float, msg: str, i: int = i) -> None:
                if progress:
                    progress((i + f) / max(1, len(pending)), msg)

            sub(0.0, f"Uploading Short #{rank}: {m['title']}")
            body = self.build_body(m, privacy, when)
            self.db.update_short_upload(job_id, rank, upload_status="uploading", upload_error=None, privacy=privacy)
            try:
                vid = self.upload_video(path, body, sub)
            except QuotaExceededError as exc:
                msg = f"YouTube quota reached ({exc}). Quota resets at midnight Pacific time — run the upload again then."
                self.log.error(msg)
                for rest in pending[i:]:
                    self.db.update_short_upload(job_id, rest["rank"], upload_status="pending", upload_error=msg[:500])
                    results.append({"rank": rest["rank"], "title": rest["title"], "upload_status": "pending",
                                    "error": msg})
                break
            except JobCancelled:
                self.db.update_short_upload(job_id, rank, upload_status="pending", upload_error="cancelled")
                raise
            except Exception as exc:
                self.log.error("Short #%d upload failed: %s", rank, exc)
                self.db.update_short_upload(job_id, rank, upload_status="failed", upload_error=str(exc)[:500])
                results.append({"rank": rank, "title": m["title"], "upload_status": "failed", "error": str(exc)})
                if isinstance(exc, NotAuthorizedError):
                    break
                continue

            notes = []
            srt = path.with_suffix(".srt")
            if self.s.youtube_upload_captions and srt.exists() and srt.stat().st_size > 0:
                try:
                    self.upload_captions(vid, srt)
                except Exception as exc:  # captions are a bonus — never fail the upload
                    notes.append(f"captions not uploaded: {exc}")
            thumb = Path(m.get("thumbnail_path") or "")
            if self.s.youtube_set_thumbnail and thumb.is_file():
                try:
                    self.set_thumbnail(vid, thumb)
                except Exception as exc:
                    notes.append(f"thumbnail not set: {exc}")
            for n in notes:
                self.log.warning("Short #%d: %s", rank, n)

            status = "scheduled" if when else "uploaded"
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self.db.update_short_upload(
                job_id, rank, youtube_video_id=vid, upload_status=status, uploaded_at=now,
                scheduled_at=when.astimezone(timezone.utc).isoformat(timespec="seconds") if when else None,
                upload_error="; ".join(notes)[:500] or None, title=m["title"], description=m.get("description", ""),
                hashtags=" ".join(m.get("hashtags", [])), tags=json.dumps(m.get("tags", []), ensure_ascii=False))
            res = {"rank": rank, "title": m["title"], "upload_status": status, "youtube_video_id": vid,
                   "url": f"https://youtube.com/shorts/{vid}", "privacy": privacy,
                   "publish_at": when.isoformat() if when else None, "notes": notes}
            results.append(res)
            self.log.info("✔ Short #%d %s: %s%s", rank, status, res["url"],
                          f" (goes public {when.astimezone():%Y-%m-%d %H:%M})" if when else f" ({privacy})")
            sub(1.0, f"Short #{rank} {status}")

        self._write_log(out_dir, results)
        return sorted(results, key=lambda r: r["rank"])

    @staticmethod
    def job_private_only(job: Dict[str, Any]) -> bool:
        try:
            return bool(json.loads(job.get("settings_json") or "{}").get("private_only"))
        except ValueError:
            return False

    @staticmethod
    def _row_result(m: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
        vid = row.get("youtube_video_id")
        return {"rank": m["rank"], "title": m["title"], "upload_status": row.get("upload_status") or "uploaded",
                "youtube_video_id": vid, "url": f"https://youtube.com/shorts/{vid}" if vid else "",
                "privacy": row.get("privacy"), "publish_at": row.get("scheduled_at")}

    @staticmethod
    def _write_log(out_dir: Path, results: List[Dict[str, Any]]) -> None:
        path = out_dir / "uploads.json"
        existing = {r["rank"]: r for r in (read_json(path) or [])}
        for r in results:
            if not r.get("skipped"):
                existing[r["rank"]] = r
        atomic_write_json(path, sorted(existing.values(), key=lambda r: r["rank"]))


def save_edited_metadata(out_dir: Path, rank: int, title: str, description: str, hashtags: List[str],
                         tags: List[str]) -> None:
    """Persist edits from the UI into metadata.json / metadata.csv and the sidecar JSON."""
    from core.models import ShortMetadata
    from modules.metadata_generator import clean_hashtags, clean_title, write_metadata_files

    metas = [ShortMetadata.model_validate(m) for m in (read_json(out_dir / "metadata.json") or [])]
    for m in metas:
        if m.rank == rank:
            m.title = clean_title(title, 100)
            m.description = description.strip()
            m.hashtags = clean_hashtags(hashtags, m.category, "")
            m.tags = clean_tags(tags)
            if m.output_path:
                sidecar = Path(m.output_path).with_suffix(".json")
                data = read_json(sidecar)
                if data:
                    data.update(title=m.title, description=m.description, hashtags=m.hashtags, tags=m.tags)
                    atomic_write_json(sidecar, data)
    write_metadata_files(out_dir, metas)
