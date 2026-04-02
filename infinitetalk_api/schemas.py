from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class GenerationOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    size: Literal["infinitetalk-480", "infinitetalk-720"] = "infinitetalk-480"
    mode: Literal["clip", "streaming"] = "streaming"
    frame_num: int = Field(default=81, ge=5)
    motion_frame: int = Field(default=9, ge=1)
    max_frame_num: int = Field(default=1000, ge=5)
    sample_steps: int = Field(default=40, ge=1)
    sample_shift: float | None = None
    sample_text_guide_scale: float = 5.0
    sample_audio_guide_scale: float = 4.0
    seed: int = 42
    color_correction_strength: float = 1.0
    scene_seg: bool = False
    use_teacache: bool = False
    teacache_thresh: float = 0.2
    use_apg: bool = False
    apg_momentum: float = -0.75
    apg_norm_threshold: float = 55.0

    @field_validator("frame_num")
    @classmethod
    def validate_frame_num(cls, value: int) -> int:
        if (value - 1) % 4 != 0:
            raise ValueError("frame_num must satisfy 4n+1.")
        return value

    @model_validator(mode="after")
    def validate_lengths(self) -> "GenerationOptions":
        if self.mode == "clip" and self.max_frame_num < self.frame_num:
            self.max_frame_num = self.frame_num
        return self


class DriverOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["upload", "tts"] = "upload"
    audio_type: Literal["add", "para"] = "para"
    tts_text: str | None = None
    human1_voice: str | None = None
    human2_voice: str | None = None

    @model_validator(mode="after")
    def validate_driver(self) -> "DriverOptions":
        if self.type == "tts" and not self.tts_text:
            raise ValueError("driver.tts_text is required when driver.type is 'tts'.")
        return self


class JobCreatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str | None = None
    negative_prompt: str = ""
    bbox: dict[str, list[float]] | None = None
    audio_paths: list[str] = Field(default_factory=list)
    generation: GenerationOptions = Field(default_factory=GenerationOptions)
    driver: DriverOptions = Field(default_factory=DriverOptions)


class AvatarRecord(BaseModel):
    id: str
    name: str
    prompt: str
    media_kind: Literal["image", "video"]
    media_path: str
    bbox: dict[str, list[float]] | None = None
    defaults: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class JobArtifacts(BaseModel):
    input_audio_paths: list[str] = Field(default_factory=list)
    source_audio_paths: list[str] = Field(default_factory=list)
    result_video_path: str | None = None
    result_audio_path: str | None = None


class JobRecord(BaseModel):
    id: str
    avatar_id: str
    status: Literal[
        "queued",
        "preprocessing",
        "generating",
        "muxing",
        "completed",
        "failed",
    ]
    request: dict[str, Any]
    artifacts: JobArtifacts = Field(default_factory=JobArtifacts)
    progress: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    created_at: str
    updated_at: str


class SystemOptionsResponse(BaseModel):
    sizes: list[str]
    modes: list[str]
    default_generation: GenerationOptions
    default_voice_1: str
    default_voice_2: str
    available_voices: list[str] = Field(default_factory=list)
    model_loaded: bool
    queue_size: int

