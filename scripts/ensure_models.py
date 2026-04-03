from __future__ import annotations

import os
import sys
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
        resume_download=True,
    )


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
        resume_download=True,
    )


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


def main() -> int:
    try:
        ensure_core_models()
        ensure_kokoro()
    except Exception as exc:
        log(str(exc))
        return 1

    log("Model bootstrap completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

