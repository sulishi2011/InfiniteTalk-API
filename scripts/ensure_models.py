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
    )


def download_snapshot_patterns(
    repo_id: str,
    local_dir: Path,
    token: str | None,
    allow_patterns: list[str],
) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    joined = ", ".join(allow_patterns)
    log(f"Downloading {repo_id} [{joined}] -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
        token=token,
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
    auto_download_all_presets = env_bool("INFINITETALK_AUTO_DOWNLOAD_ALL_ACCEL_PRESETS", False)
    token = token_from_env()
    default_preset = os.getenv("INFINITETALK_DEFAULT_PRESET", "base").strip().lower()
    if default_preset in {"quality", "balanced", "fast"}:
        required_presets = (
            {"quality", "balanced", "fast"}
            if auto_download and auto_download_all_presets
            else {default_preset}
        )
    else:
        required_presets = set()
    if not required_presets:
        return

    bundle_repo = os.getenv(
        "INFINITETALK_LIGHTX2V_BUNDLE_REPO",
        "lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v",
    )
    audio_encoder_repo = os.getenv(
        "INFINITETALK_LIGHTX2V_AUDIO_ENCODER_REPO",
        "TencentGameMate/chinese-hubert-large",
    )
    audio_encoder_dir = Path(
        os.getenv(
            "INFINITETALK_LIGHTX2V_AUDIO_ENCODER_DIR",
            "/workspace/weights/TencentGameMate-chinese-hubert-large",
        )
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
    preset_requirements = {
        "quality": {
            "required": [
                "config.json",
                "google/umt5-xxl/tokenizer.json",
                "xlm-roberta-large/tokenizer.json",
                "distill_models/distill_model.safetensors",
                "distill_models/models_t5_umt5-xxl-enc-bf16.pth",
                "distill_models/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                "distill_models/Wan2.1_VAE.pth",
            ],
            "download_patterns": [
                "config.json",
                "google/**",
                "xlm-roberta-large/**",
                "distill_models/**",
            ],
        },
        "balanced": {
            "required": [
                "config.json",
                "google/umt5-xxl/tokenizer.json",
                "xlm-roberta-large/tokenizer.json",
                "distill_fp8/non_block.safetensors",
                "distill_fp8/models_t5_umt5-xxl-enc-fp8.pth",
                "distill_fp8/clip-fp8.pth",
                "distill_fp8/Wan2.1_VAE.pth",
            ],
            "download_patterns": [
                "config.json",
                "google/**",
                "xlm-roberta-large/**",
                "distill_fp8/**",
            ],
        },
        "fast": {
            "required": [
                "config.json",
                "google/umt5-xxl/tokenizer.json",
                "xlm-roberta-large/tokenizer.json",
                "distill_int8/non_block.safetensors",
                "distill_int8/models_t5_umt5-xxl-enc-int8.pth",
                "distill_int8/clip-int8.pth",
                "distill_int8/Wan2.1_VAE.pth",
            ],
            "download_patterns": [
                "config.json",
                "google/**",
                "xlm-roberta-large/**",
                "distill_int8/**",
            ],
        },
    }

    missing_messages: list[str] = []
    for preset in sorted(required_presets):
        model_dir = preset_model_dirs[preset]
        requirement = preset_requirements[preset]
        required_paths = [model_dir / rel_path for rel_path in requirement["required"]]
        missing_required = [path for path in required_paths if not path.exists()]
        if missing_required:
            if not auto_download:
                missing_messages.append(
                    f"missing {', '.join(str(path) for path in missing_required)} for preset '{preset}'"
                )
            else:
                if model_dir.exists() and not model_dir.is_dir():
                    missing_messages.append(f"{model_dir} for preset '{preset}' is not a directory")
                    continue
                download_snapshot_patterns(
                    bundle_repo,
                    model_dir,
                    token,
                    requirement["download_patterns"],
                )
                missing_required = [path for path in required_paths if not path.exists()]
                if missing_required:
                    missing_messages.append(
                        f"downloaded {bundle_repo}, but still missing {', '.join(str(path) for path in missing_required)} for preset '{preset}'"
                    )

    audio_encoder_required = [
        audio_encoder_dir / "config.json",
        audio_encoder_dir / "preprocessor_config.json",
        audio_encoder_dir / "chinese-hubert-large-fairseq-ckpt.pt",
    ]
    missing_audio_encoder = [path for path in audio_encoder_required if not path.exists()]
    if missing_audio_encoder:
        if not auto_download:
            missing_messages.append(
                "missing LightX2V audio encoder files: "
                + ", ".join(str(path) for path in missing_audio_encoder)
            )
        else:
            download_snapshot(audio_encoder_repo, audio_encoder_dir, token)
            missing_audio_encoder = [path for path in audio_encoder_required if not path.exists()]
            if missing_audio_encoder:
                missing_messages.append(
                    "downloaded LightX2V audio encoder repo, but still missing "
                    + ", ".join(str(path) for path in missing_audio_encoder)
                )

    if missing_messages:
        joined = "; ".join(missing_messages)
        raise FileNotFoundError(
            "Accelerated preset models are missing. "
            f"{joined}. Mount the official LightX2V StepDistill model roots plus the "
            "TencentGameMate audio encoder into the container, or set "
            "INFINITETALK_AUTO_DOWNLOAD_ACCEL_MODELS=true so startup can download the "
            "matching preset assets automatically."
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
