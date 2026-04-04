from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

from wan.utils.multitalk_utils import save_video_ffmpeg


@dataclass(slots=True)
class LightX2VPresetSpec:
    preset: str
    model_path: str
    lightx2v_root: str
    attn_mode: str
    output_fps: int
    target_fps: int
    video_duration: int
    offload_granularity: str
    cpu_offload: bool
    text_encoder_offload: bool
    image_encoder_offload: bool
    vae_offload: bool
    audio_encoder_offload: bool
    audio_adapter_offload: bool
    use_tiling_vae: bool
    use_31_block: bool
    feature_caching: str
    teacache_thresh: float | None
    dit_quantized: bool
    text_encoder_quantized: bool
    image_encoder_quantized: bool
    adapter_quantized: bool
    quant_scheme: str | None


class LightX2VBackend:
    def __init__(self, spec: LightX2VPresetSpec):
        self.spec = spec
        self._pipe = self._build_pipeline(spec)

    def _build_pipeline(self, spec: LightX2VPresetSpec):
        root = Path(spec.lightx2v_root).expanduser()
        if not root.exists():
            raise FileNotFoundError(
                f"INFINITETALK_LIGHTX2V_ROOT points to a missing directory: {root}"
            )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        os.environ.setdefault("DTYPE", "BF16")
        os.environ.setdefault("SENSITIVE_LAYER_DTYPE", "None")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("PLATFORM", "cuda")

        from lightx2v import LightX2VPipeline
        from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
        from lightx2v.utils.utils import seed_all

        self._init_empty_input_info = init_empty_input_info
        self._update_input_info_from_dict = update_input_info_from_dict
        self._seed_all = seed_all

        pipe = LightX2VPipeline(
            model_path=spec.model_path,
            model_cls="seko_talk",
            task="s2v",
        )
        pipe.target_fps = spec.target_fps
        pipe.audio_sr = 16000
        pipe.video_duration = spec.video_duration
        pipe.use_31_block = spec.use_31_block
        pipe.offload_granularity = spec.offload_granularity
        pipe.audio_encoder_cpu_offload = spec.audio_encoder_offload
        pipe.audio_adapter_cpu_offload = spec.audio_adapter_offload
        pipe.use_tiling_vae = spec.use_tiling_vae
        pipe.adapter_quantized = spec.adapter_quantized
        pipe.adapter_quant_scheme = spec.quant_scheme

        pipe.enable_offload(
            cpu_offload=spec.cpu_offload,
            offload_granularity=spec.offload_granularity,
            text_encoder_offload=spec.text_encoder_offload,
            image_encoder_offload=spec.image_encoder_offload,
            vae_offload=spec.vae_offload,
        )
        if spec.dit_quantized:
            pipe.enable_quantize(
                dit_quantized=True,
                text_encoder_quantized=spec.text_encoder_quantized,
                image_encoder_quantized=spec.image_encoder_quantized,
                quant_scheme=spec.quant_scheme or "fp8-q8f",
            )
        if spec.feature_caching != "NoCaching":
            pipe.enable_cache(
                cache_method=spec.feature_caching,
                teacache_thresh=spec.teacache_thresh or 0.25,
            )

        pipe.create_generator(
            attn_mode=spec.attn_mode,
            infer_steps=4,
            num_frames=81,
            height=480,
            width=832,
            guidance_scale=1.0,
            sample_shift=5.0,
            fps=spec.output_fps,
        )
        return pipe

    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        image_path: str,
        audio_path: str,
        generation: Any,
        output_prefix: Path,
    ) -> dict[str, str]:
        fixed_area = "720p" if generation.size == "infinitetalk-720" else "480p"
        self._pipe.infer_steps = generation.sample_steps
        self._pipe.target_video_length = generation.frame_num
        self._pipe.modify_config(
            {
                "infer_steps": generation.sample_steps,
                "target_video_length": generation.frame_num,
                "sample_shift": generation.sample_shift or 5.0,
                "sample_guide_scale": generation.sample_text_guide_scale,
                "enable_cfg": generation.sample_text_guide_scale != 1.0,
                "resize_mode": "keep_ratio_fixed_area",
                "fixed_area": fixed_area,
            }
        )
        self._pipe.target_video_length = generation.frame_num

        result = self._run_pipeline(
            seed=generation.seed,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_path=image_path,
            audio_path=audio_path,
            save_result_path=str(output_prefix) + ".mp4",
            return_result_tensor=True,
        )
        if not isinstance(result, dict):
            raise RuntimeError("LightX2V returned an unexpected result payload.")

        video_tensor = result.get("video")
        audio_tensor = result.get("audio")
        if video_tensor is None or audio_tensor is None:
            raise RuntimeError(
                "LightX2V did not return video/audio tensors. "
                "Check that the SekoTalk runner and weights are configured correctly."
            )

        return self._save_result(
            video_tensor=video_tensor,
            audio_tensor=audio_tensor,
            output_prefix=output_prefix,
            fps=self.spec.output_fps,
        )

    def _run_pipeline(
        self,
        *,
        seed: int,
        prompt: str,
        negative_prompt: str,
        image_path: str,
        audio_path: str,
        save_result_path: str,
        return_result_tensor: bool,
    ) -> Any:
        self._pipe.seed = seed
        self._pipe.prompt = prompt
        self._pipe.negative_prompt = negative_prompt
        self._pipe.image_path = image_path
        self._pipe.audio_path = audio_path
        self._pipe.save_result_path = save_result_path
        self._pipe.return_result_tensor = return_result_tensor

        input_info = self._init_empty_input_info(self._pipe.task, self._pipe.support_tasks)
        self._seed_all(seed)
        self._update_input_info_from_dict(input_info, self._pipe)
        return self._pipe.runner.run_pipeline(input_info)

    def _save_result(
        self,
        *,
        video_tensor: torch.Tensor,
        audio_tensor: dict[str, Any],
        output_prefix: Path,
        fps: int,
    ) -> dict[str, str]:
        waveform = audio_tensor.get("waveform")
        sample_rate = int(audio_tensor.get("sample_rate", 16000))
        if waveform is None:
            raise RuntimeError("LightX2V audio payload is missing waveform data.")

        temp_audio_path = output_prefix.parent / f"{output_prefix.name}_audio.wav"
        waveform_np = waveform.squeeze().detach().cpu().numpy()
        sf.write(temp_audio_path, waveform_np, sample_rate)

        video_chtw = video_tensor.permute(3, 0, 1, 2).detach().cpu()
        video_chtw = video_chtw.mul(2.0).sub(1.0).clamp(-1.0, 1.0)
        save_video_ffmpeg(
            video_chtw,
            str(output_prefix),
            [str(temp_audio_path)],
            fps=fps,
            high_quality_save=False,
        )
        result_video_path = str(output_prefix) + ".mp4"
        return {
            "result_video_path": result_video_path,
            "result_audio_path": str(temp_audio_path),
        }
