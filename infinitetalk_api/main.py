from __future__ import annotations

import json
import logging
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from .common import deep_merge, utc_now_iso
from .config import ServiceConfig
from .repository import AvatarRepository, JobRepository
from .runtime import InfiniteTalkRuntime
from .schemas import AvatarRecord, JobArtifacts, JobCreatePayload, JobRecord, SystemOptionsResponse
from .worker import JobWorker

LOGGER = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def _save_upload(upload: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as file_obj:
        shutil.copyfileobj(upload.file, file_obj)


def _copy_local_file(source: Path, destination: Path) -> None:
    if not source.exists():
        raise HTTPException(status_code=400, detail=f"Local path does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _parse_optional_json(raw_value: str | None, field_name: str):
    if raw_value is None or not raw_value.strip():
        return None
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} must be valid JSON.") from exc


def _list_available_voices(config: ServiceConfig) -> list[str]:
    voices_dir = Path(config.kokoro_dir) / "voices"
    if not voices_dir.exists():
        return []
    return sorted(str(path) for path in voices_dir.glob("*.pt"))


def _validate_avatar_defaults(defaults: dict | None) -> None:
    if defaults is None:
        return
    merged = deep_merge(defaults, {"driver": {"type": "upload"}, "audio_paths": ["/tmp/fake.wav"]})
    try:
        JobCreatePayload.model_validate(merged)
    except Exception:
        # Defaults are partial by design. Only reject obviously invalid top-level types.
        if not isinstance(defaults, dict):
            raise HTTPException(status_code=400, detail="defaults_json must decode to a JSON object.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="InfiniteTalk API",
        version="0.1.0",
        description="Avatar-based API wrapper for InfiniteTalk generation.",
    )

    @app.on_event("startup")
    def startup() -> None:
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
        config = ServiceConfig.from_env()
        config.ensure_directories()
        avatars = AvatarRepository(config.avatars_dir)
        jobs = JobRepository(config.jobs_dir)
        jobs.mark_incomplete_jobs_failed()
        runtime = InfiniteTalkRuntime(config)
        worker = JobWorker(runtime, avatars, jobs)
        worker.start()
        if config.startup_load_model:
            runtime.ensure_loaded()

        app.state.config = config
        app.state.avatars = avatars
        app.state.jobs = jobs
        app.state.runtime = runtime
        app.state.worker = worker

    @app.on_event("shutdown")
    def shutdown() -> None:
        worker: JobWorker | None = getattr(app.state, "worker", None)
        if worker is not None:
            worker.stop()

    @app.get("/healthz")
    def healthz(request: Request) -> dict:
        runtime: InfiniteTalkRuntime = request.app.state.runtime
        worker: JobWorker = request.app.state.worker
        return {
            "status": "ok",
            "model_loaded": runtime.model_loaded,
            "queue_size": worker.queue_size(),
        }

    @app.get("/v1/system/options", response_model=SystemOptionsResponse)
    def get_system_options(request: Request) -> SystemOptionsResponse:
        config: ServiceConfig = request.app.state.config
        runtime: InfiniteTalkRuntime = request.app.state.runtime
        worker: JobWorker = request.app.state.worker
        return SystemOptionsResponse(
            sizes=["infinitetalk-480", "infinitetalk-720"],
            modes=["clip", "streaming"],
            default_generation={
                "size": config.default_size,
                "mode": config.default_mode,
                "frame_num": config.default_frame_num,
                "motion_frame": config.default_motion_frame,
                "max_frame_num": config.default_max_frame_num,
                "sample_steps": config.default_sample_steps,
                "sample_text_guide_scale": config.default_text_guide_scale,
                "sample_audio_guide_scale": config.default_audio_guide_scale,
                "color_correction_strength": config.default_color_correction_strength,
            },
            default_voice_1=config.default_voice_1,
            default_voice_2=config.default_voice_2,
            available_voices=_list_available_voices(config),
            model_loaded=runtime.model_loaded,
            queue_size=worker.queue_size(),
        )

    @app.get("/v1/avatars", response_model=list[AvatarRecord])
    def list_avatars(request: Request) -> list[AvatarRecord]:
        avatars: AvatarRepository = request.app.state.avatars
        return avatars.list()

    @app.get("/v1/avatars/{avatar_id}", response_model=AvatarRecord)
    def get_avatar(avatar_id: str, request: Request) -> AvatarRecord:
        avatars: AvatarRepository = request.app.state.avatars
        avatar = avatars.get(avatar_id)
        if avatar is None:
            raise HTTPException(status_code=404, detail=f"Avatar {avatar_id} not found.")
        return avatar

    @app.post("/v1/avatars", response_model=AvatarRecord)
    async def create_avatar(
        request: Request,
        name: str = Form(...),
        prompt: str = Form(...),
        media_file: UploadFile | None = File(default=None),
        media_path: str | None = Form(default=None),
        bbox_json: str | None = Form(default=None),
        defaults_json: str | None = Form(default=None),
    ) -> AvatarRecord:
        config: ServiceConfig = request.app.state.config
        avatars: AvatarRepository = request.app.state.avatars

        if media_file is None and not media_path:
            raise HTTPException(status_code=400, detail="Provide media_file or media_path.")
        if media_file is not None and media_path:
            raise HTTPException(status_code=400, detail="Use either media_file or media_path, not both.")

        source_path = Path(media_path) if media_path else None
        file_name = media_file.filename if media_file is not None else source_path.name if source_path else ""
        extension = Path(file_name or "").suffix.lower()
        if extension not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Input media must be an image or video.")

        bbox = _parse_optional_json(bbox_json, "bbox_json")
        defaults = _parse_optional_json(defaults_json, "defaults_json") or {}
        if defaults and not isinstance(defaults, dict):
            raise HTTPException(status_code=400, detail="defaults_json must decode to a JSON object.")
        _validate_avatar_defaults(defaults)

        avatar_id = uuid.uuid4().hex
        avatar_dir = config.avatars_dir / avatar_id
        media_path = avatar_dir / f"source{extension}"
        if media_file is not None:
            _save_upload(media_file, media_path)
        else:
            _copy_local_file(source_path, media_path)  # type: ignore[arg-type]

        record = AvatarRecord(
            id=avatar_id,
            name=name,
            prompt=prompt,
            media_kind="video" if extension in VIDEO_EXTENSIONS else "image",
            media_path=str(media_path),
            bbox=bbox,
            defaults=defaults,
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
        )
        return avatars.create(record)

    @app.get("/v1/jobs", response_model=list[JobRecord])
    def list_jobs(request: Request) -> list[JobRecord]:
        jobs: JobRepository = request.app.state.jobs
        return jobs.list()

    @app.get("/v1/jobs/{job_id}", response_model=JobRecord)
    def get_job(job_id: str, request: Request) -> JobRecord:
        jobs: JobRepository = request.app.state.jobs
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
        return job

    @app.post("/v1/jobs", response_model=JobRecord)
    async def create_job(
        request: Request,
        avatar_id: str = Form(...),
        request_json: str = Form(...),
        audio_1: UploadFile | None = File(default=None),
        audio_2: UploadFile | None = File(default=None),
        audio_path_1: str | None = Form(default=None),
        audio_path_2: str | None = Form(default=None),
    ) -> JobRecord:
        config: ServiceConfig = request.app.state.config
        avatars: AvatarRepository = request.app.state.avatars
        jobs: JobRepository = request.app.state.jobs
        worker: JobWorker = request.app.state.worker

        avatar = avatars.get(avatar_id)
        if avatar is None:
            raise HTTPException(status_code=404, detail=f"Avatar {avatar_id} not found.")

        try:
            raw_request = json.loads(request_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="request_json must be valid JSON.") from exc

        if not isinstance(raw_request, dict):
            raise HTTPException(status_code=400, detail="request_json must decode to a JSON object.")

        effective_request = deep_merge(avatar.defaults, raw_request)
        try:
            JobCreatePayload.model_validate(effective_request)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        job_id = uuid.uuid4().hex
        job_dir = config.jobs_dir / job_id
        inputs_dir = job_dir / "inputs"
        input_audio_paths: list[str] = []
        uploads = [audio_1, audio_2]
        for index, upload in enumerate(uploads, start=1):
            if upload is None or not upload.filename:
                continue
            extension = Path(upload.filename).suffix.lower()
            destination = inputs_dir / f"audio_{index}{extension}"
            _save_upload(upload, destination)
            input_audio_paths.append(str(destination))

        local_audio_paths = [audio_path_1, audio_path_2]
        for index, raw_path in enumerate(local_audio_paths, start=1):
            if not raw_path:
                continue
            source_path = Path(raw_path)
            extension = source_path.suffix.lower()
            destination = inputs_dir / f"audio_local_{index}{extension}"
            _copy_local_file(source_path, destination)
            input_audio_paths.append(str(destination))

        driver_type = effective_request.get("driver", {}).get("type", "upload")
        if len(input_audio_paths) > 2:
            raise HTTPException(
                status_code=400,
                detail="At most two input audio tracks are supported per job.",
            )
        if driver_type == "upload" and not input_audio_paths and not effective_request.get("audio_paths"):
            raise HTTPException(
                status_code=400,
                detail="Upload driver requires audio_1/audio_2 files or request_json.audio_paths.",
            )

        job = JobRecord(
            id=job_id,
            avatar_id=avatar_id,
            status="queued",
            request=raw_request,
            artifacts=JobArtifacts(input_audio_paths=input_audio_paths),
            progress={"stage": "queued"},
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
        )
        jobs.create(job)
        worker.enqueue(job_id)
        return jobs.get(job_id)  # type: ignore[return-value]

    @app.get("/v1/jobs/{job_id}/result")
    def get_job_result(job_id: str, request: Request) -> dict:
        jobs: JobRepository = request.app.state.jobs
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
        if job.status != "completed":
            raise HTTPException(status_code=409, detail=f"Job {job_id} is not completed yet.")
        return {
            "job_id": job.id,
            "status": job.status,
            "result_video_path": job.artifacts.result_video_path,
            "result_audio_path": job.artifacts.result_audio_path,
            "source_audio_paths": job.artifacts.source_audio_paths,
        }

    @app.get("/v1/jobs/{job_id}/video")
    def download_result_video(job_id: str, request: Request) -> FileResponse:
        jobs: JobRepository = request.app.state.jobs
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
        if job.status != "completed" or not job.artifacts.result_video_path:
            raise HTTPException(status_code=409, detail=f"Job {job_id} has no completed video output.")
        result_path = Path(job.artifacts.result_video_path)
        if not result_path.exists():
            raise HTTPException(status_code=404, detail="Result video file is missing.")
        return FileResponse(result_path, media_type="video/mp4", filename=result_path.name)

    return app


app = create_app()
