from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import uuid


QUEUE_JOB_TYPES = {"training", "predicting", "test_split_eval"}
QUEUE_FINAL_STATUSES = {"completed", "failed", "cancelled", "skipped"}
QUEUE_ACTIVE_STATUSES = {"queued", "running", "waiting_on_parent"}


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_queue_job(
    *,
    job_type: str,
    title: str,
    source_tab: str,
    config_snapshot: dict[str, object],
    summary_text: str,
    parent_job_id: str | None = None,
    status: str = "queued",
) -> dict[str, object]:
    if job_type not in QUEUE_JOB_TYPES:
        raise ValueError(f"Unsupported queue job type: {job_type}")
    return {
        "job_id": uuid.uuid4().hex[:10],
        "job_type": job_type,
        "status": status,
        "title": title,
        "created_at": now_iso_utc(),
        "source_tab": source_tab,
        "config_snapshot": deepcopy(config_snapshot),
        "artifacts": {},
        "error_message": None,
        "parent_job_id": parent_job_id,
        "summary_text": summary_text,
    }


def clone_queue_job(job: dict[str, object]) -> dict[str, object]:
    duplicated = deepcopy(job)
    duplicated["job_id"] = uuid.uuid4().hex[:10]
    duplicated["status"] = "queued"
    duplicated["created_at"] = now_iso_utc()
    duplicated["artifacts"] = {}
    duplicated["error_message"] = None
    duplicated["parent_job_id"] = None
    return duplicated


def format_queue_job_label(job: dict[str, object]) -> str:
    status = str(job.get("status", "queued"))
    job_type = str(job.get("job_type", "unknown"))
    title = str(job.get("title", "Untitled Job"))
    summary = str(job.get("summary_text", "")).strip()
    parent_job_id = job.get("parent_job_id")
    first_line = f"[{status}] {job_type} | {title}"
    if parent_job_id:
        first_line += f" | parent={parent_job_id}"
    return first_line if not summary else f"{first_line}\n{summary}"


def is_terminal_status(status: str) -> bool:
    return status in QUEUE_FINAL_STATUSES
