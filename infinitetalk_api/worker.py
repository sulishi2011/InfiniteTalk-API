from __future__ import annotations

import logging
import queue
import threading

from .repository import AvatarRepository, JobRepository
from .runtime import InfiniteTalkRuntime

LOGGER = logging.getLogger(__name__)


class JobWorker:
    def __init__(
        self,
        runtime: InfiniteTalkRuntime,
        avatars: AvatarRepository,
        jobs: JobRepository,
    ):
        self.runtime = runtime
        self.avatars = avatars
        self.jobs = jobs
        self._queue: "queue.Queue[str | None]" = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="infinitetalk-job-worker", daemon=True)

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._queue.put(None)
        self._thread.join(timeout=5)

    def enqueue(self, job_id: str) -> None:
        self._queue.put(job_id)

    def queue_size(self) -> int:
        return self._queue.qsize()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                job_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if job_id is None:
                self._queue.task_done()
                break

            try:
                self._process_job(job_id)
            except Exception:
                LOGGER.exception("Unhandled error in job worker for %s", job_id)
            finally:
                self._queue.task_done()

    def _process_job(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if job is None:
            LOGGER.warning("Job %s disappeared before execution.", job_id)
            return
        avatar = self.avatars.get(job.avatar_id)
        if avatar is None:
            self.jobs.update(
                job_id,
                status="failed",
                error=f"Avatar {job.avatar_id} not found.",
            )
            return

        def on_status(status: str, progress: dict | None = None) -> None:
            self.jobs.update(job_id, status=status, progress=progress or {})

        try:
            result = self.runtime.generate_job(job, avatar, status_callback=on_status)
            self.jobs.update(
                job_id,
                status="completed",
                progress={"stage": "done"},
                artifacts={
                    **job.artifacts.model_dump(mode="json"),
                    **result,
                },
                error=None,
            )
        except Exception as exc:
            LOGGER.exception("Job %s failed.", job_id)
            self.jobs.update(
                job_id,
                status="failed",
                error=str(exc),
            )

