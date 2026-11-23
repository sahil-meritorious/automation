"""Background job execution for the Streamlit UI (survives page reruns)."""
from __future__ import annotations

import threading
from typing import Dict, Optional

from config import Settings
from database.database import get_database

_jobs: Dict[str, Dict] = {}
_lock = threading.Lock()
_startup_done = False


def startup_cleanup(settings: Settings) -> None:
    """Once per process: jobs marked 'running' from a previous process become 'interrupted'."""
    global _startup_done
    with _lock:
        if _startup_done:
            return
        _startup_done = True
    get_database(settings.db_path).mark_interrupted(exclude_ids=list(_jobs))


def start_job(source: str, settings: Settings, job_id: Optional[str] = None) -> str:
    from core.pipeline import ShortsPipeline  # local import keeps UI start-up fast

    db = get_database(settings.db_path)
    if job_id is None:
        job_id = db.create_job(source, settings.public_dict())
    cancel = threading.Event()
    pipeline = ShortsPipeline(settings, db=db, cancel_event=cancel)

    def _target() -> None:
        try:
            pipeline.run(source, job_id=job_id)
        finally:
            with _lock:
                _jobs.pop(job_id, None)

    thread = threading.Thread(target=_target, name=f"job-{job_id}", daemon=True)
    with _lock:
        _jobs[job_id] = {"thread": thread, "cancel": cancel, "source": source}
    thread.start()
    return job_id


def is_running(job_id: str) -> bool:
    with _lock:
        job = _jobs.get(job_id)
    return bool(job and job["thread"].is_alive())


def cancel_job(job_id: str) -> bool:
    with _lock:
        job = _jobs.get(job_id)
    if job:
        job["cancel"].set()
        return True
    return False


def running_jobs() -> list:
    with _lock:
        return [jid for jid, j in _jobs.items() if j["thread"].is_alive()]
