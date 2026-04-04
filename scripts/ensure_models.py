from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def token_from_env() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


def log(message: str) -> None:
    print(f"[ensure_models] {message}", flush=True)


def require_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def download_snapshot(repo_id: str, local_dir: Path, token: str | None) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    log(f"Downloading snapshot {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        token=token,
    )


def download_snapshot_subdir(
    repo_id: str,
    remote_subdir: str,
    target_dir: Path,
    token: str | None,
) -> None:
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    log(f"Downloading {repo_id}:{remote_subdir}/ -> {target_dir}")
    with tempfile.TemporaryDirectory(dir=str(target_dir.parent)) as tmp_dir:
        tmp_path = Path(tmp_dir)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(tmp_path),
            allow_patterns=[f"{remote_subdir}/**"],
            token=token,
        )
        repo_name = repo_id.split("/")[-1]
        candidates = [
            tmp_path / remote_subdir,
            tmp_path / repo_name / remote_subdir,
        ]
        source_dir = next((candidate for candidate in candidates if candidate.exists()), None)
        if source_dir is None:
            raise FileNotFoundError(
                f"Downloaded repo {repo_id}, but could not find subdir {remote_subdir}."
            )
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.move(str(source_dir), str(target_dir))


def download_file(
    repo_id: str,
    filename: str,
    local_dir: Path,
    token: str | None,
    revision: str | None = None,
) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    log(f"Downloading {repo_id}:{filename} -> {local_dir}")
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_dir=str(local_dir),
        token=token,
    )


def normalize_downloaded_file(local_path: Path, filename: str) -> None:
    nested_candidate = local_path.parent / Path(filename)
    if local_path.exists() or not nested_candidate.exists():
        return
    require_parent(local_path)
    nested_candidate.replace(local_path)


def ensure_core_models() -> None:
    auto_download = env_bool("INFINITETALK_AUTO_DOWNLOAD_MODELS", True)
    token = token_from_env()

    ckpt_dir = Path(os.getenv("INFINITETALK_CKPT_DIR", "/workspace/weights/Wan2.1-I2V-14B-480P"))
    wav2vec_dir = Path(os.getenv("INFINITETALK_WAV2VEC_DIR", "/workspace/weights/chinese-wav2vec2-base"))
    infinitetalk_path = Path(
        os.getenv(
            "INFINITETALK_MODEL_PATH",
            "/workspace/weights/InfiniteTalk/single/infinitetalk.safetensors",
        )
    )

    missing_messages: list[str] = []

    if not ckpt_dir.exists():
        if not auto_download:
            missing_messages.append(f"missing {ckpt_dir}")
        else:
            download_snapshot(
                os.getenv("INFINITETALK_BASE_REPO", "Wan-AI/Wan2.1-I2V-14B-480P"),
                ckpt_dir,
                token,
            )

    if not wav2vec_dir.exists():
        if not auto_download:
            missing_messages.append(f"missing {wav2vec_dir}")
        else:
            download_snapshot(
                os.getenv("INFINITETALK_WAV2VEC_REPO", "TencentGameMate/chinese-wav2vec2-base"),
                wav2vec_dir,
                token,
            )

    wav2vec_model_file = wav2vec_dir / "model.safetensors"
    if not wav2vec_model_file.exists():
        if not auto_download:
            missing_messages.append(f"missing {wav2vec_model_file}")
        else:
            download_file(
                os.getenv("INFINITETALK_WAV2VEC_REPO", "TencentGameMate/chinese-wav2vec2-base"),
                "model.safetensors",
                wav2vec_dir,
                token,
                revision=os.getenv("INFINITETALK_WAV2VEC_REVISION", "refs/pr/1"),
            )

    if not infinitetalk_path.exists():
        if not auto_download:
            missing_messages.append(f"missing {infinitetalk_path}")
        else:
            require_parent(infinitetalk_path)
            download_snapshot(
                os.getenv("INFINITETALK_FINETUNE_REPO", "MeiGen-AI/InfiniteTalk"),
                infinitetalk_path.parent.parent,
                token,
            )

    if missing_messages:
        joined = "; ".join(missing_messages)
        raise FileNotFoundError(
            "Core InfiniteTalk models are missing. "
            f"{joined}. Mount the weights into the container or set "
            "INFINITETALK_AUTO_DOWNLOAD_MODELS=true."
        )


def ensure_kokoro() -> None:
    auto_download = env_bool("INFINITETALK_AUTO_DOWNLOAD_KOKORO", False)
    if not auto_download:
        return

    token = token_from_env()
    kokoro_dir = Path(os.getenv("INFINITETALK_KOKORO_DIR", "/workspace/weights/Kokoro-82M"))
    if kokoro_dir.exists():
        return

    download_snapshot(
        os.getenv("INFINITETALK_KOKORO_REPO", "hexgrad/Kokoro-82M"),
        kokoro_dir,
        token,
    )


def ensure_accelerated_models() -> None:
    auto_download = env_bool("INFINITETALK_AUTO_DOWNLOAD_ACCEL_MODELS", False)
    token = token_from_env()
    default_preset = os.getenv("INFINITETALK_DEFAULT_PRESET", "base").strip().lower()
    required_presets = {default_preset} if default_preset in {"quality", "balanced", "fast"} else set()
    if not required_presets:
        return

    bundle_repo = os.getenv(
        "INFINITETALK_LIGHTX2V_BUNDLE_REPO",
        "lightx2v/Wan2.1-Distill-Models",
    )

    preset_model_dirs = {
        "quality": Path(
            os.getenv(
                "INFINITETALK_LIGHTX2V_QUALITY_MODEL_DIR",
                "/workspace/weights/SekoTalk-Distill",
            )
        ),
        "balanced": Path(
            os.getenv(
                "INFINITETALK_LIGHTX2V_BALANCED_MODEL_DIR",
                "/workspace/weights/SekoTalk-Distill-fp8",
            )
        ),
        "fast": Path(
            os.getenv(
                "INFINITETALK_LIGHTX2V_FAST_MODEL_DIR",
                "/workspace/weights/SekoTalk-Distill-int8",
            )
        ),
    }
    preset_remote_subdirs = {
        "quality": os.getenv(
            "INFINITETALK_LIGHTX2V_QUALITY_BUNDLE_SUBDIR",
            "SekoTalk-Distill",
        ),
        "balanced": os.getenv(
            "INFINITETALK_LIGHTX2V_BALANCED_BUNDLE_SUBDIR",
            "SekoTalk-Distill-fp8",
        ),
        "fast": os.getenv(
            "INFINITETALK_LIGHTX2V_FAST_BUNDLE_SUBDIR",
            "SekoTalk-Distill-int8",
        ),
    }

    missing_messages: list[str] = []
    for preset in sorted(required_presets):
        model_dir = preset_model_dirs[preset]
        remote_subdir = preset_remote_subdirs[preset]
        if model_dir.exists():
            if not model_dir.is_dir():
                missing_messages.append(f"{model_dir} for preset '{preset}' is not a directory")
                continue
            if any(model_dir.iterdir()):
                continue
            if not auto_download:
                missing_messages.append(f"empty model directory {model_dir} for preset '{preset}'")
                continue
            shutil.rmtree(model_dir)

        if not auto_download:
            missing_messages.append(f"missing {model_dir} for preset '{preset}'")
            continue

        download_snapshot_subdir(
            bundle_repo,
            remote_subdir,
            model_dir,
            token,
        )

    if missing_messages:
        joined = "; ".join(missing_messages)
        raise FileNotFoundError(
            "Accelerated preset model bundles are missing. "
            f"{joined}. Mount the official LightX2V SekoTalk bundle directories into the "
            "container, or set INFINITETALK_AUTO_DOWNLOAD_ACCEL_MODELS=true so startup can "
            "download the matching preset bundle automatically."
        )


def main() -> int:
    try:
        ensure_core_models()
        ensure_accelerated_models()
        ensure_kokoro()
    except Exception as exc:
        log(str(exc))
        return 1

    log("Model bootstrap completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
