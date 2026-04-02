from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterable

from .common import utc_now_iso
from .schemas import AvatarRecord, JobRecord


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


class AvatarRepository:
    def __init__(self, root: Path):
        self.root = root
        self._lock = threading.Lock()

    def _record_path(self, avatar_id: str) -> Path:
        return self.root / avatar_id / "avatar.json"

    def create(self, record: AvatarRecord) -> AvatarRecord:
        with self._lock:
            _write_json(self._record_path(record.id), record.model_dump(mode="json"))
        return record

    def get(self, avatar_id: str) -> AvatarRecord | None:
        path = self._record_path(avatar_id)
        if not path.exists():
            return None
        return AvatarRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[AvatarRecord]:
        records: list[AvatarRecord] = []
        for path in sorted(self.root.glob("*/avatar.json")):
            records.append(AvatarRecord.model_validate_json(path.read_text(encoding="utf-8")))
        records.sort(key=lambda item: item.created_at, reverse=True)
        return records


class JobRepository:
    def __init__(self, root: Path):
        self.root = root
        self._lock = threading.Lock()

    def _record_path(self, job_id: str) -> Path:
        return self.root / job_id / "job.json"

    def create(self, record: JobRecord) -> JobRecord:
        with self._lock:
            _write_json(self._record_path(record.id), record.model_dump(mode="json"))
        return record

    def get(self, job_id: str) -> JobRecord | None:
        path = self._record_path(job_id)
        if not path.exists():
            return None
        return JobRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[JobRecord]:
        records: list[JobRecord] = []
        for path in sorted(self.root.glob("*/job.json")):
            records.append(JobRecord.model_validate_json(path.read_text(encoding="utf-8")))
        records.sort(key=lambda item: item.created_at, reverse=True)
        return records

    def update(self, job_id: str, **changes) -> JobRecord:
        with self._lock:
            record = self.get(job_id)
            if record is None:
                raise KeyError(f"Job {job_id} not found.")
            merged = record.model_dump(mode="json")
            for key, value in changes.items():
                merged[key] = value
            merged["updated_at"] = utc_now_iso()
            updated = JobRecord.model_validate(merged)
            _write_json(self._record_path(job_id), updated.model_dump(mode="json"))
            return updated

    def mark_incomplete_jobs_failed(self) -> Iterable[JobRecord]:
        updated_records: list[JobRecord] = []
        for record in self.list():
            if record.status in {"queued", "preprocessing", "generating", "muxing"}:
                updated_records.append(
                    self.update(
                        record.id,
                        status="failed",
                        error="Service restarted before the job finished.",
                    )
                )
        return updated_records

