from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
from einops import rearrange
from kokoro import KPipeline
from transformers import Wav2Vec2FeatureExtractor

import wan
from src.audio_analysis.wav2vec2 import Wav2Vec2Model
from wan.configs import WAN_CONFIGS
from wan.utils.multitalk_utils import save_video_ffmpeg
from wan.utils.segvideo import shot_detect
from wan.utils.utils import is_video, split_wav_librosa

from .common import deep_merge
from .config import ServiceConfig
from .schemas import AvatarRecord, JobCreatePayload, JobRecord

StatusCallback = Callable[[str, dict[str, Any] | None], None]

LOGGER = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def _default_sample_shift(size: str) -> float:
    if size == "infinitetalk-480":
        return 7.0
    if size == "infinitetalk-720":
        return 11.0
    raise ValueError(f"Unsupported size: {size}")


@dataclass(slots=True)
class PipelineSpec:
    key: str
    checkpoint_dir: str
    infinitetalk_dir: str
    quant_dir: str | None
    dit_path: str | None
    lora_dirs: list[str] | None
    lora_scales: list[float] | None
    quant: str | None
    t5_checkpoint_path: str | None


class InfiniteTalkRuntime:
    def __init__(self, config: ServiceConfig):
        self.config = config
        self._pipeline = None
        self._pipeline_key: str | None = None
        self._wav2vec_feature_extractor = None
        self._audio_encoder = None
        self._tts_pipeline = None
        self._load_lock = threading.Lock()

    @property
    def model_loaded(self) -> bool:
        return self._pipeline is not None

    def ensure_loaded(self, preset: str | None = None) -> None:
        desired_preset = preset or self.config.default_preset
        with self._load_lock:
            world_size = int(os.getenv("WORLD_SIZE", "1"))
            if world_size != 1:
                raise RuntimeError("The API service currently supports single-process, single-GPU deployment only.")

            self._ensure_shared_components_loaded()
            pipeline_spec = self._resolve_pipeline_spec(desired_preset)
            if self._pipeline is not None and self._pipeline_key == pipeline_spec.key:
                return

            self._unload_pipeline()
            self._pipeline = self._build_pipeline(pipeline_spec)
            self._pipeline_key = pipeline_spec.key

    def _ensure_shared_components_loaded(self) -> None:
        if self._wav2vec_feature_extractor is not None and self._audio_encoder is not None:
            return

        wav2vec_dir = self._resolve_existing_dir(
            self.config.wav2vec_dir,
            "INFINITETALK_WAV2VEC_DIR",
        )
        ckpt_dir = self._resolve_existing_dir(
            self.config.ckpt_dir,
            "INFINITETALK_CKPT_DIR",
        )
        infinitetalk_path = self._resolve_existing_file(
            self.config.infinitetalk_dir,
            "INFINITETALK_MODEL_PATH",
        )
        if Path(self.config.kokoro_dir).exists():
            self.config.kokoro_dir = str(Path(self.config.kokoro_dir).resolve())

        self.config.wav2vec_dir = str(wav2vec_dir)
        self.config.ckpt_dir = str(ckpt_dir)
        self.config.infinitetalk_dir = str(infinitetalk_path)

        for attr_name, env_name in (
            ("distilled_t5_path", "INFINITETALK_DISTILLED_T5_PATH"),
            ("distilled_infinitetalk_dir", "INFINITETALK_DISTILLED_MODEL_PATH"),
            ("quality_dit_path", "INFINITETALK_QUALITY_DIT_PATH"),
            ("balanced_dit_path", "INFINITETALK_BALANCED_DIT_PATH"),
            ("fast_dit_path", "INFINITETALK_FAST_DIT_PATH"),
        ):
            raw_value = getattr(self.config, attr_name)
            if raw_value:
                setattr(
                    self.config,
                    attr_name,
                    str(self._resolve_existing_file(raw_value, env_name)),
                )

        LOGGER.info("Loading wav2vec audio encoder from %s", self.config.wav2vec_dir)
        self._wav2vec_feature_extractor, self._audio_encoder = self._custom_init(
            "cpu",
            self.config.wav2vec_dir,
        )

    def _resolve_pipeline_spec(self, preset: str) -> PipelineSpec:
        base_patch = self.config.infinitetalk_dir
        distilled_patch = self.config.distilled_infinitetalk_dir or base_patch

        if preset == "base":
            return PipelineSpec(
                key="base",
                checkpoint_dir=self.config.ckpt_dir,
                infinitetalk_dir=base_patch,
                quant_dir=self.config.quant_dir,
                dit_path=self.config.dit_path,
                lora_dirs=self.config.lora_dirs,
                lora_scales=self.config.lora_scales,
                quant=self.config.quant,
                t5_checkpoint_path=None,
            )

        dit_path = {
            "quality": self.config.quality_dit_path,
            "balanced": self.config.balanced_dit_path,
            "fast": self.config.fast_dit_path,
        }.get(preset)
        if not dit_path:
            raise FileNotFoundError(
                f"Preset '{preset}' is not configured. Set the matching distilled model path env var."
            )

        return PipelineSpec(
            key=preset,
            checkpoint_dir=self.config.ckpt_dir,
            infinitetalk_dir=distilled_patch,
            quant_dir=None,
            dit_path=dit_path,
            lora_dirs=None,
            lora_scales=None,
            quant=None,
            t5_checkpoint_path=self.config.distilled_t5_path,
        )

    def _build_pipeline(self, pipeline_spec: PipelineSpec):
        cfg = WAN_CONFIGS[self.config.task]
        device_id = int(os.getenv("LOCAL_RANK", "0"))
        LOGGER.info(
            "Loading InfiniteTalk pipeline preset '%s' from %s",
            pipeline_spec.key,
            pipeline_spec.checkpoint_dir,
        )
        lora_dirs = pipeline_spec.lora_dirs
        lora_scales = pipeline_spec.lora_scales
        if lora_dirs is not None and lora_scales is None:
            lora_scales = [1.0] * len(lora_dirs)

        pipeline = wan.InfiniteTalkPipeline(
            config=cfg,
            checkpoint_dir=pipeline_spec.checkpoint_dir,
            quant_dir=pipeline_spec.quant_dir,
            device_id=device_id,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=False,
            lora_dir=lora_dirs,
            lora_scales=lora_scales,
            quant=pipeline_spec.quant,
            dit_path=pipeline_spec.dit_path,
            infinitetalk_dir=pipeline_spec.infinitetalk_dir,
            t5_checkpoint_path=pipeline_spec.t5_checkpoint_path,
        )
        if self.config.num_persistent_param_in_dit is not None:
            pipeline.vram_management = True
            pipeline.enable_vram_management(
                num_persistent_param_in_dit=self.config.num_persistent_param_in_dit
            )
        return pipeline

    def _unload_pipeline(self) -> None:
        if self._pipeline is None:
            return
        self._pipeline = None
        self._pipeline_key = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _resolve_existing_dir(self, raw_path: str, env_name: str) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"{env_name} points to a missing directory: {path}. "
                "Check your Runpod volume mount path and weights layout."
            )
        if not path.is_dir():
            raise NotADirectoryError(f"{env_name} must be a directory: {path}")
        return path

    def _resolve_existing_file(self, raw_path: str, env_name: str) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"{env_name} points to a missing file: {path}. "
                "Check your Runpod volume mount path and weights layout."
            )
        if not path.is_file():
            raise FileNotFoundError(f"{env_name} must be a file: {path}")
        return path

    def generate_job(
        self,
        job: JobRecord,
        avatar: AvatarRecord,
        status_callback: StatusCallback | None = None,
    ) -> dict[str, Any]:
        merged_request = self._apply_generation_preset_defaults(
            deep_merge(avatar.defaults, job.request)
        )
        payload = JobCreatePayload.model_validate(merged_request)
        self.ensure_loaded(payload.generation.preset)
        generation = payload.generation
        prompt = payload.prompt or avatar.prompt
        bbox = payload.bbox or avatar.bbox
        job_dir = self.config.jobs_dir / job.id
        source_dir = job_dir / "source_audio"
        scene_dir = job_dir / "scene_segments"
        embed_dir = job_dir / "embeddings"
        output_dir = job_dir / "outputs"
        for path in (source_dir, scene_dir, embed_dir, output_dir):
            path.mkdir(parents=True, exist_ok=True)

        if status_callback:
            status_callback("preprocessing", {"stage": "prepare_audio"})
        source_audio_paths, final_audio_path = self._prepare_source_audio(
            payload,
            job,
            source_dir,
        )

        cond_lists = self._build_condition_lists(
            media_path=avatar.media_path,
            source_audio_paths=source_audio_paths,
            scene_seg=generation.scene_seg,
            scene_dir=scene_dir,
        )

        generated_list = []
        total_segments = len(cond_lists[0])
        extras = self._build_generation_extras(generation)
        for index, items in enumerate(zip(*cond_lists), start=1):
            if status_callback:
                status_callback(
                    "generating",
                    {
                        "stage": "generate_segment",
                        "segment_index": index,
                        "segment_total": total_segments,
                    },
                )
            input_clip = {
                "prompt": prompt,
                "cond_video": items[0],
                "cond_audio": {},
            }
            if bbox is not None:
                input_clip["bbox"] = bbox
            if len(source_audio_paths) == 2:
                input_clip["audio_type"] = payload.driver.audio_type

            segment_embed_dir = embed_dir / f"segment_{index:03d}"
            segment_embed_dir.mkdir(parents=True, exist_ok=True)
            cond_audio, segment_audio_path = self._prepare_segment_embeddings(
                audio_paths=list(items[1:]),
                audio_type=payload.driver.audio_type,
                embed_dir=segment_embed_dir,
                frame_num=generation.frame_num,
            )
            input_clip["cond_audio"] = cond_audio
            input_clip["video_audio"] = str(segment_audio_path)

            video = self._pipeline.generate_infinitetalk(
                input_clip,
                size_buckget=generation.size,
                motion_frame=generation.motion_frame,
                frame_num=generation.frame_num,
                shift=generation.sample_shift or _default_sample_shift(generation.size),
                sampling_steps=generation.sample_steps,
                text_guide_scale=generation.sample_text_guide_scale,
                audio_guide_scale=generation.sample_audio_guide_scale,
                n_prompt=payload.negative_prompt,
                seed=generation.seed,
                offload_model=self.config.offload_model,
                max_frames_num=(
                    generation.frame_num
                    if generation.mode == "clip"
                    else generation.max_frame_num
                ),
                color_correction_strength=generation.color_correction_strength,
                extra_args=extras,
            )
            generated_list.append(video)

        if status_callback:
            status_callback("muxing", {"stage": "mux_audio_video"})
        output_prefix = output_dir / "result"
        sum_video = torch.cat(generated_list, dim=1)
        save_video_ffmpeg(
            sum_video,
            str(output_prefix),
            [str(final_audio_path)],
            high_quality_save=False,
        )
        result_video_path = str(output_prefix) + ".mp4"
        return {
            "result_video_path": result_video_path,
            "result_audio_path": str(final_audio_path),
            "source_audio_paths": [str(path) for path in source_audio_paths],
        }

    def _apply_generation_preset_defaults(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        generation_payload = request_payload.setdefault("generation", {})
        if not isinstance(generation_payload, dict):
            return request_payload

        preset = generation_payload.get("preset", self.config.default_preset)
        generation_payload.setdefault("preset", preset)
        preset_defaults = self._preset_defaults(preset)
        for key, value in preset_defaults.items():
            generation_payload.setdefault(key, value)
        return request_payload

    def _preset_defaults(self, preset: str) -> dict[str, Any]:
        if preset == "quality":
            return {
                "size": "infinitetalk-480",
                "mode": "clip",
                "frame_num": 81,
                "sample_steps": 4,
                "sample_shift": 2.0,
                "sample_text_guide_scale": 1.0,
                "sample_audio_guide_scale": 2.0,
                "use_teacache": False,
            }
        if preset == "balanced":
            return {
                "size": "infinitetalk-480",
                "mode": "clip",
                "frame_num": 81,
                "sample_steps": 4,
                "sample_shift": 2.0,
                "sample_text_guide_scale": 1.0,
                "sample_audio_guide_scale": 2.0,
                "use_teacache": True,
                "teacache_thresh": 0.25,
            }
        if preset == "fast":
            return {
                "size": "infinitetalk-480",
                "mode": "clip",
                "frame_num": 41,
                "sample_steps": 4,
                "sample_shift": 2.0,
                "sample_text_guide_scale": 1.0,
                "sample_audio_guide_scale": 2.0,
                "use_teacache": True,
                "teacache_thresh": 0.3,
            }
        return {}

    def _build_generation_extras(self, generation) -> SimpleNamespace:
        return SimpleNamespace(
            use_teacache=generation.use_teacache,
            teacache_thresh=generation.teacache_thresh,
            size=generation.size,
            use_apg=generation.use_apg,
            apg_momentum=generation.apg_momentum,
            apg_norm_threshold=generation.apg_norm_threshold,
        )

    def _build_condition_lists(
        self,
        media_path: str,
        source_audio_paths: list[Path],
        scene_seg: bool,
        scene_dir: Path,
    ) -> list[list[str]]:
        cond_lists: list[list[str]] = []
        if scene_seg and is_video(media_path):
            time_list, cond_list = shot_detect(media_path, str(scene_dir))
            if len(time_list) == 0:
                cond_lists.append([media_path])
                for audio_path in source_audio_paths:
                    cond_lists.append([str(audio_path)])
            else:
                cond_lists.append(cond_list)
                for audio_path in source_audio_paths:
                    cond_lists.append(split_wav_librosa(str(audio_path), time_list, str(scene_dir)))
            return cond_lists

        cond_lists.append([media_path])
        for audio_path in source_audio_paths:
            cond_lists.append([str(audio_path)])
        return cond_lists

    def _prepare_source_audio(
        self,
        payload: JobCreatePayload,
        job: JobRecord,
        source_dir: Path,
    ) -> tuple[list[Path], Path]:
        if payload.driver.type == "tts":
            if payload.driver.human2_voice:
                return self._process_tts_multi(
                    payload.driver.tts_text or "",
                    source_dir,
                    payload.driver.human1_voice or self.config.default_voice_1,
                    payload.driver.human2_voice,
                )
            return self._process_tts_single(
                payload.driver.tts_text or "",
                source_dir,
                payload.driver.human1_voice or self.config.default_voice_1,
            )

        input_audio_paths = job.artifacts.input_audio_paths or payload.audio_paths
        if len(input_audio_paths) not in {1, 2}:
            raise ValueError("Upload driver requires one or two audio inputs.")

        copied_audio_paths: list[Path] = []
        for index, audio_path in enumerate(input_audio_paths, start=1):
            source_path = Path(audio_path)
            if not source_path.exists():
                raise FileNotFoundError(f"Audio file not found: {source_path}")
            destination = source_dir / f"speaker_{index}{source_path.suffix.lower()}"
            if source_path.resolve() != destination.resolve():
                shutil.copy2(source_path, destination)
            else:
                destination = source_path
            copied_audio_paths.append(destination)

        if len(copied_audio_paths) == 1:
            human_speech = self._audio_prepare_single(copied_audio_paths[0])
            final_audio_path = source_dir / "sum_all.wav"
            sf.write(final_audio_path, human_speech, 16000)
            return copied_audio_paths, final_audio_path

        _, _, sum_human_speechs = self._audio_prepare_multi(
            copied_audio_paths[0],
            copied_audio_paths[1],
            payload.driver.audio_type,
        )
        final_audio_path = source_dir / "sum_all.wav"
        sf.write(final_audio_path, sum_human_speechs, 16000)
        return copied_audio_paths, final_audio_path

    def _prepare_segment_embeddings(
        self,
        audio_paths: list[str],
        audio_type: str,
        embed_dir: Path,
        frame_num: int,
    ) -> tuple[dict[str, str], Path]:
        cond_audio: dict[str, str] = {}
        if len(audio_paths) == 1:
            human_speech = self._audio_prepare_single(Path(audio_paths[0]))
            audio_embedding = self._get_embedding(
                human_speech,
                self._wav2vec_feature_extractor,
                self._audio_encoder,
            )
            self._validate_embedding_length(
                audio_embedding=audio_embedding,
                frame_num=frame_num,
                speaker_label="person1",
            )
            emb_path = embed_dir / "person1.pt"
            sum_audio = embed_dir / "sum.wav"
            sf.write(sum_audio, human_speech, 16000)
            torch.save(audio_embedding, emb_path)
            cond_audio["person1"] = str(emb_path)
            return cond_audio, sum_audio

        new_human_speech1, new_human_speech2, sum_human_speechs = self._audio_prepare_multi(
            Path(audio_paths[0]),
            Path(audio_paths[1]),
            audio_type,
        )
        audio_embedding_1 = self._get_embedding(
            new_human_speech1,
            self._wav2vec_feature_extractor,
            self._audio_encoder,
        )
        self._validate_embedding_length(
            audio_embedding=audio_embedding_1,
            frame_num=frame_num,
            speaker_label="person1",
        )
        audio_embedding_2 = self._get_embedding(
            new_human_speech2,
            self._wav2vec_feature_extractor,
            self._audio_encoder,
        )
        self._validate_embedding_length(
            audio_embedding=audio_embedding_2,
            frame_num=frame_num,
            speaker_label="person2",
        )
        emb1_path = embed_dir / "person1.pt"
        emb2_path = embed_dir / "person2.pt"
        sum_audio = embed_dir / "sum.wav"
        sf.write(sum_audio, sum_human_speechs, 16000)
        torch.save(audio_embedding_1, emb1_path)
        torch.save(audio_embedding_2, emb2_path)
        cond_audio["person1"] = str(emb1_path)
        cond_audio["person2"] = str(emb2_path)
        return cond_audio, sum_audio

    def _custom_init(self, device: str, wav2vec_dir: str):
        audio_encoder = Wav2Vec2Model.from_pretrained(
            wav2vec_dir,
            local_files_only=True,
        ).to(device)
        audio_encoder.feature_extractor._freeze_parameters()
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            wav2vec_dir,
            local_files_only=True,
        )
        return wav2vec_feature_extractor, audio_encoder

    def _validate_embedding_length(
        self,
        audio_embedding: torch.Tensor,
        frame_num: int,
        speaker_label: str,
    ) -> None:
        embedding_steps = int(audio_embedding.shape[0])
        if embedding_steps > frame_num:
            return

        min_audio_seconds = (frame_num + 1) / 25.0
        actual_audio_seconds = embedding_steps / 25.0
        raise ValueError(
            f"Audio for {speaker_label} is too short for frame_num={frame_num}. "
            f"Need embedding steps > {frame_num}, got {embedding_steps} "
            f"(about {actual_audio_seconds:.2f}s at 25 fps). "
            f"Use a longer audio file or lower generation.frame_num to <= {max(5, embedding_steps - 1)}. "
            f"Minimum audio length for this frame_num is about {min_audio_seconds:.2f}s."
        )

    def _loudness_norm(self, audio_array, sr: int = 16000, lufs: int = -23):
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio_array)
        if abs(loudness) > 100:
            return audio_array
        return pyln.normalize.loudness(audio_array, loudness, lufs)

    def _extract_audio_from_video(self, filename: Path, sample_rate: int) -> np.ndarray:
        raw_audio_path = filename.parent / f"{filename.stem}_{uuid.uuid4().hex}.wav"
        ffmpeg_command = [
            "ffmpeg",
            "-y",
            "-i",
            str(filename),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "2",
            str(raw_audio_path),
        ]
        subprocess.run(ffmpeg_command, check=True)
        try:
            human_speech_array, sr = librosa.load(raw_audio_path, sr=sample_rate)
            return self._loudness_norm(human_speech_array, sr)
        finally:
            if raw_audio_path.exists():
                raw_audio_path.unlink()

    def _audio_prepare_single(self, audio_path: Path, sample_rate: int = 16000):
        ext = audio_path.suffix.lower()
        if ext in VIDEO_EXTENSIONS:
            return self._extract_audio_from_video(audio_path, sample_rate)
        human_speech_array, sr = librosa.load(audio_path, sr=sample_rate)
        return self._loudness_norm(human_speech_array, sr)

    def _audio_prepare_multi(
        self,
        left_path: Path,
        right_path: Path,
        audio_type: str,
        sample_rate: int = 16000,
    ):
        human_speech_array1 = self._audio_prepare_single(left_path, sample_rate)
        human_speech_array2 = self._audio_prepare_single(right_path, sample_rate)
        if audio_type == "para":
            max_length = max(human_speech_array1.shape[0], human_speech_array2.shape[0])
            new_human_speech1 = np.pad(
                human_speech_array1,
                (0, max_length - human_speech_array1.shape[0]),
            )
            new_human_speech2 = np.pad(
                human_speech_array2,
                (0, max_length - human_speech_array2.shape[0]),
            )
        elif audio_type == "add":
            new_human_speech1 = np.concatenate(
                [
                    human_speech_array1[: human_speech_array1.shape[0]],
                    np.zeros(human_speech_array2.shape[0]),
                ]
            )
            new_human_speech2 = np.concatenate(
                [
                    np.zeros(human_speech_array1.shape[0]),
                    human_speech_array2[: human_speech_array2.shape[0]],
                ]
            )
        else:
            raise ValueError(f"Unsupported audio_type: {audio_type}")
        sum_human_speechs = new_human_speech1 + new_human_speech2
        return new_human_speech1, new_human_speech2, sum_human_speechs

    def _get_tts_pipeline(self):
        if self._tts_pipeline is None:
            self._tts_pipeline = KPipeline(lang_code="a", repo_id=self.config.kokoro_dir)
        return self._tts_pipeline

    def _process_tts_single(
        self,
        text: str,
        save_dir: Path,
        voice1: str,
    ) -> tuple[list[Path], Path]:
        pipeline = self._get_tts_pipeline()
        voice_tensor = torch.load(voice1, weights_only=True)
        generator = pipeline(text, voice=voice_tensor, speed=1, split_pattern=r"\n+")
        audios = []
        for _, _, audio in generator:
            audios.append(audio)
        merged = torch.concat(audios, dim=0)
        speaker_path = save_dir / "speaker_1.wav"
        sf.write(speaker_path, merged, 24000)
        return [speaker_path], speaker_path

    def _process_tts_multi(
        self,
        text: str,
        save_dir: Path,
        voice1: str,
        voice2: str,
    ) -> tuple[list[Path], Path]:
        pattern = r"\(s(\d+)\)\s*(.*?)(?=\s*\(s\d+\)|$)"
        matches = re.findall(pattern, text, re.DOTALL)
        if not matches:
            raise ValueError("Multi-speaker TTS text must use the '(s1)...(s2)...' format.")

        pipeline = self._get_tts_pipeline()
        s1_sentences = []
        s2_sentences = []
        for speaker, content in matches:
            if speaker == "1":
                voice_tensor = torch.load(voice1, weights_only=True)
                generator = pipeline(content, voice=voice_tensor, speed=1, split_pattern=r"\n+")
                audios = [audio for _, _, audio in generator]
                merged = torch.concat(audios, dim=0)
                s1_sentences.append(merged)
                s2_sentences.append(torch.zeros_like(merged))
            elif speaker == "2":
                voice_tensor = torch.load(voice2, weights_only=True)
                generator = pipeline(content, voice=voice_tensor, speed=1, split_pattern=r"\n+")
                audios = [audio for _, _, audio in generator]
                merged = torch.concat(audios, dim=0)
                s2_sentences.append(merged)
                s1_sentences.append(torch.zeros_like(merged))

        speaker1 = torch.concat(s1_sentences, dim=0)
        speaker2 = torch.concat(s2_sentences, dim=0)
        merged = speaker1 + speaker2

        speaker1_path = save_dir / "speaker_1.wav"
        speaker2_path = save_dir / "speaker_2.wav"
        sum_path = save_dir / "sum.wav"
        sf.write(speaker1_path, speaker1, 24000)
        sf.write(speaker2_path, speaker2, 24000)
        sf.write(sum_path, merged, 24000)
        return [speaker1_path, speaker2_path], sum_path

    def _get_embedding(
        self,
        speech_array,
        wav2vec_feature_extractor,
        audio_encoder,
        sr: int = 16000,
        device: str = "cpu",
    ):
        audio_duration = len(speech_array) / sr
        video_length = audio_duration * 25
        audio_feature = np.squeeze(
            wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
        )
        audio_feature = torch.from_numpy(audio_feature).float().to(device=device).unsqueeze(0)
        with torch.no_grad():
            embeddings = audio_encoder(
                audio_feature,
                seq_len=int(video_length),
                output_hidden_states=True,
            )
        if len(embeddings) == 0:
            raise RuntimeError("Failed to extract audio embedding.")
        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d")
        return audio_emb.cpu().detach()
