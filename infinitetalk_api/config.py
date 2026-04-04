from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str) -> list[str] | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_float_list(name: str) -> list[float] | None:
    values = _env_list(name)
    if values is None:
        return None
    return [float(value) for value in values]


@dataclass(slots=True)
class ServiceConfig:
    host: str
    port: int
    data_root: Path
    avatars_dir: Path
    jobs_dir: Path
    task: str
    ckpt_dir: str
    infinitetalk_dir: str
    wav2vec_dir: str
    kokoro_dir: str
    quant_dir: str | None
    dit_path: str | None
    lora_dirs: list[str] | None
    lora_scales: list[float] | None
    quant: str | None
    distilled_t5_path: str | None
    distilled_infinitetalk_dir: str | None
    quality_dit_path: str | None
    balanced_dit_path: str | None
    fast_dit_path: str | None
    lightx2v_root: str
    lightx2v_attn_mode: str
    lightx2v_output_fps: int
    lightx2v_target_fps: int
    lightx2v_video_duration: int
    lightx2v_offload_granularity: str
    lightx2v_cpu_offload: bool
    lightx2v_text_encoder_offload: bool
    lightx2v_image_encoder_offload: bool
    lightx2v_vae_offload: bool
    lightx2v_audio_encoder_offload: bool
    lightx2v_audio_adapter_offload: bool
    lightx2v_use_tiling_vae: bool
    lightx2v_quality_model_dir: str | None
    lightx2v_balanced_model_dir: str | None
    lightx2v_fast_model_dir: str | None
    offload_model: bool
    num_persistent_param_in_dit: int | None
    default_preset: str
    default_size: str
    default_mode: str
    default_frame_num: int
    default_max_frame_num: int
    default_motion_frame: int
    default_sample_steps: int
    default_text_guide_scale: float
    default_audio_guide_scale: float
    default_color_correction_strength: float
    default_voice_1: str
    default_voice_2: str
    startup_load_model: bool

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        data_root = Path(os.getenv("INFINITETALK_DATA_ROOT", "./runtime_data")).resolve()
        offload_model = _env_bool("INFINITETALK_OFFLOAD_MODEL", True)
        return cls(
            host=os.getenv("INFINITETALK_API_HOST", "0.0.0.0"),
            port=int(os.getenv("INFINITETALK_API_PORT", "8000")),
            data_root=data_root,
            avatars_dir=data_root / "avatars",
            jobs_dir=data_root / "jobs",
            task=os.getenv("INFINITETALK_TASK", "infinitetalk-14B"),
            ckpt_dir=os.getenv("INFINITETALK_CKPT_DIR", "./weights/Wan2.1-I2V-14B-480P"),
            infinitetalk_dir=os.getenv(
                "INFINITETALK_MODEL_PATH",
                "./weights/InfiniteTalk/single/infinitetalk.safetensors",
            ),
            wav2vec_dir=os.getenv("INFINITETALK_WAV2VEC_DIR", "./weights/chinese-wav2vec2-base"),
            kokoro_dir=os.getenv("INFINITETALK_KOKORO_DIR", "./weights/Kokoro-82M"),
            quant_dir=os.getenv("INFINITETALK_QUANT_DIR") or None,
            dit_path=os.getenv("INFINITETALK_DIT_PATH") or None,
            lora_dirs=_env_list("INFINITETALK_LORA_DIRS"),
            lora_scales=_env_float_list("INFINITETALK_LORA_SCALES"),
            quant=os.getenv("INFINITETALK_QUANT") or None,
            distilled_t5_path=os.getenv("INFINITETALK_DISTILLED_T5_PATH") or None,
            distilled_infinitetalk_dir=os.getenv("INFINITETALK_DISTILLED_MODEL_PATH") or None,
            quality_dit_path=os.getenv("INFINITETALK_QUALITY_DIT_PATH") or None,
            balanced_dit_path=os.getenv("INFINITETALK_BALANCED_DIT_PATH") or None,
            fast_dit_path=os.getenv("INFINITETALK_FAST_DIT_PATH") or None,
            lightx2v_root=os.getenv("INFINITETALK_LIGHTX2V_ROOT", "/opt/LightX2V"),
            lightx2v_attn_mode=os.getenv("INFINITETALK_LIGHTX2V_ATTN_MODE", "flash_attn2"),
            lightx2v_output_fps=int(os.getenv("INFINITETALK_LIGHTX2V_OUTPUT_FPS", "16")),
            lightx2v_target_fps=int(os.getenv("INFINITETALK_LIGHTX2V_TARGET_FPS", "16")),
            lightx2v_video_duration=int(os.getenv("INFINITETALK_LIGHTX2V_VIDEO_DURATION", "360")),
            lightx2v_offload_granularity=os.getenv(
                "INFINITETALK_LIGHTX2V_OFFLOAD_GRANULARITY",
                "block",
            ),
            lightx2v_cpu_offload=_env_bool("INFINITETALK_LIGHTX2V_CPU_OFFLOAD", offload_model),
            lightx2v_text_encoder_offload=_env_bool(
                "INFINITETALK_LIGHTX2V_TEXT_ENCODER_OFFLOAD",
                False,
            ),
            lightx2v_image_encoder_offload=_env_bool(
                "INFINITETALK_LIGHTX2V_IMAGE_ENCODER_OFFLOAD",
                False,
            ),
            lightx2v_vae_offload=_env_bool("INFINITETALK_LIGHTX2V_VAE_OFFLOAD", False),
            lightx2v_audio_encoder_offload=_env_bool(
                "INFINITETALK_LIGHTX2V_AUDIO_ENCODER_OFFLOAD",
                False,
            ),
            lightx2v_audio_adapter_offload=_env_bool(
                "INFINITETALK_LIGHTX2V_AUDIO_ADAPTER_OFFLOAD",
                False,
            ),
            lightx2v_use_tiling_vae=_env_bool("INFINITETALK_LIGHTX2V_USE_TILING_VAE", True),
            lightx2v_quality_model_dir=os.getenv(
                "INFINITETALK_LIGHTX2V_QUALITY_MODEL_DIR",
                "/workspace/weights/SekoTalk-Distill",
            )
            or None,
            lightx2v_balanced_model_dir=os.getenv(
                "INFINITETALK_LIGHTX2V_BALANCED_MODEL_DIR",
                "/workspace/weights/SekoTalk-Distill-fp8",
            )
            or None,
            lightx2v_fast_model_dir=os.getenv(
                "INFINITETALK_LIGHTX2V_FAST_MODEL_DIR",
                "/workspace/weights/SekoTalk-Distill-int8",
            )
            or None,
            offload_model=offload_model,
            num_persistent_param_in_dit=(
                int(os.getenv("INFINITETALK_NUM_PERSISTENT_PARAM_IN_DIT"))
                if os.getenv("INFINITETALK_NUM_PERSISTENT_PARAM_IN_DIT")
                else None
            ),
            default_preset=os.getenv("INFINITETALK_DEFAULT_PRESET", "base"),
            default_size=os.getenv("INFINITETALK_DEFAULT_SIZE", "infinitetalk-480"),
            default_mode=os.getenv("INFINITETALK_DEFAULT_MODE", "streaming"),
            default_frame_num=int(os.getenv("INFINITETALK_DEFAULT_FRAME_NUM", "81")),
            default_max_frame_num=int(os.getenv("INFINITETALK_DEFAULT_MAX_FRAME_NUM", "1000")),
            default_motion_frame=int(os.getenv("INFINITETALK_DEFAULT_MOTION_FRAME", "9")),
            default_sample_steps=int(os.getenv("INFINITETALK_DEFAULT_SAMPLE_STEPS", "40")),
            default_text_guide_scale=float(
                os.getenv("INFINITETALK_DEFAULT_TEXT_GUIDE_SCALE", "5.0")
            ),
            default_audio_guide_scale=float(
                os.getenv("INFINITETALK_DEFAULT_AUDIO_GUIDE_SCALE", "4.0")
            ),
            default_color_correction_strength=float(
                os.getenv("INFINITETALK_DEFAULT_COLOR_CORRECTION_STRENGTH", "1.0")
            ),
            default_voice_1=os.getenv(
                "INFINITETALK_DEFAULT_VOICE_1",
                "./weights/Kokoro-82M/voices/am_adam.pt",
            ),
            default_voice_2=os.getenv(
                "INFINITETALK_DEFAULT_VOICE_2",
                "./weights/Kokoro-82M/voices/af_heart.pt",
            ),
            startup_load_model=_env_bool("INFINITETALK_STARTUP_LOAD_MODEL", False),
        )

    def ensure_directories(self) -> None:
        for path in (self.data_root, self.avatars_dir, self.jobs_dir):
            path.mkdir(parents=True, exist_ok=True)
