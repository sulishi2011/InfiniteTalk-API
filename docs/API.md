# InfiniteTalk API

## Overview

This service wraps the existing InfiniteTalk inference code as an avatar-based API.

- `Avatar`: stores a reusable prompt plus a reference image or reference video.
- `Job`: submits audio or TTS text to drive that avatar and produce a new talking video.
- Runtime model: single-process, single-GPU worker with a FIFO job queue.

## Endpoints

### `GET /healthz`

Returns service status, model-loaded state, and current queue size.

### `GET /v1/system/options`

Returns supported output sizes, modes, default generation parameters, and detectable Kokoro voice files.

### `POST /v1/avatars`

Creates an avatar template.

Content type: `multipart/form-data`

Fields:

- `name`: avatar name.
- `prompt`: default prompt for this avatar.
- `media_file`: reference image or reference video.
- `media_path`: optional container-local path to a reference image or video. Use this instead of `media_file` when your assets are already mounted into the container.
- `bbox_json`: optional JSON object for multi-person masks.
- `defaults_json`: optional JSON object. This is merged into future `request_json` payloads before validation.

Example:

```bash
curl -X POST http://127.0.0.1:8000/v1/avatars \
  -F 'name=demo-avatar' \
  -F 'prompt=A woman is talking in a studio.' \
  -F 'media_file=@examples/single/ref_image.png' \
  -F 'defaults_json={"generation":{"size":"infinitetalk-480","mode":"streaming"}}'
```

### `GET /v1/avatars`

Lists all avatars.

### `GET /v1/avatars/{avatar_id}`

Reads a single avatar record.

### `POST /v1/jobs`

Creates a generation job.

Content type: `multipart/form-data`

Fields:

- `avatar_id`: required avatar id.
- `request_json`: required JSON payload.
- `audio_1`: optional uploaded driver audio.
- `audio_2`: optional uploaded driver audio for multi-speaker mode.
- `audio_path_1`: optional container-local path for speaker 1 audio.
- `audio_path_2`: optional container-local path for speaker 2 audio.

The service supports two driver modes:

1. `upload`
2. `tts`

`request_json` schema:

```json
{
  "prompt": "optional prompt override",
  "negative_prompt": "",
  "bbox": {
    "person1": [10, 20, 220, 320],
    "person2": [15, 340, 225, 620]
  },
  "audio_paths": [],
  "driver": {
    "type": "upload",
    "audio_type": "para",
    "tts_text": null,
    "human1_voice": null,
    "human2_voice": null
  },
  "generation": {
    "size": "infinitetalk-480",
    "mode": "streaming",
    "frame_num": 81,
    "motion_frame": 9,
    "max_frame_num": 1000,
    "sample_steps": 40,
    "sample_shift": null,
    "sample_text_guide_scale": 5.0,
    "sample_audio_guide_scale": 4.0,
    "seed": 42,
    "color_correction_strength": 1.0,
    "scene_seg": false,
    "use_teacache": false,
    "teacache_thresh": 0.2,
    "use_apg": false,
    "apg_momentum": -0.75,
    "apg_norm_threshold": 55.0
  }
}
```

Single-speaker upload example:

```bash
curl -X POST http://127.0.0.1:8000/v1/jobs \
  -F "avatar_id=YOUR_AVATAR_ID" \
  -F 'request_json={
    "driver":{"type":"upload"},
    "generation":{"size":"infinitetalk-480","mode":"streaming","sample_steps":40}
  }' \
  -F 'audio_1=@examples/single/1.wav'
```

Multi-speaker upload example:

```bash
curl -X POST http://127.0.0.1:8000/v1/jobs \
  -F "avatar_id=YOUR_AVATAR_ID" \
  -F 'request_json={
    "driver":{"type":"upload","audio_type":"para"},
    "generation":{"size":"infinitetalk-480","mode":"streaming"}
  }' \
  -F 'audio_1=@examples/multi/1-man.WAV' \
  -F 'audio_2=@examples/multi/1-woman.WAV'
```

TTS example:

```bash
curl -X POST http://127.0.0.1:8000/v1/jobs \
  -F "avatar_id=YOUR_AVATAR_ID" \
  -F 'request_json={
    "driver":{
      "type":"tts",
      "tts_text":"Hello from InfiniteTalk.",
      "human1_voice":"./weights/Kokoro-82M/voices/am_adam.pt"
    },
    "generation":{"size":"infinitetalk-480","mode":"streaming"}
  }'
```

### `GET /v1/jobs`

Lists all jobs.

### `GET /v1/jobs/{job_id}`

Returns the full job record, including status, progress, artifact paths, and errors.

### `GET /v1/jobs/{job_id}/result`

Returns the generated video path and final audio path after the job is completed.

### `GET /v1/jobs/{job_id}/video`

Downloads the final `mp4` file for a completed job.

## Notes

- The queue is in-memory. If the service restarts, unfinished jobs are marked as failed.
- Current API deployment target is single-process, single-GPU.
- The API reuses the project's existing abilities:
  - image-to-video
  - video dubbing
  - single speaker and two-speaker mode
  - local audio uploads
  - Kokoro-based TTS
  - scene segmentation for reference videos
  - TeaCache / APG / guidance scale controls

## Docker

Build:

```bash
docker build -t infinitetalk-api:latest .
```

Run:

```bash
docker run --gpus all --rm \
  -p 8000:8000 \
  -v /path/to/weights:/app/weights \
  -v /path/to/runtime_data:/app/runtime_data \
  -e INFINITETALK_CKPT_DIR=/app/weights/Wan2.1-I2V-14B-480P \
  -e INFINITETALK_WAV2VEC_DIR=/app/weights/chinese-wav2vec2-base \
  -e INFINITETALK_MODEL_PATH=/app/weights/InfiniteTalk/single/infinitetalk.safetensors \
  -e INFINITETALK_KOKORO_DIR=/app/weights/Kokoro-82M \
  -e INFINITETALK_STARTUP_LOAD_MODEL=true \
  infinitetalk-api:latest
```

Auto-download models on first startup:

```bash
docker run --gpus all --rm \
  -p 8000:8000 \
  -v /path/to/runtime_data:/workspace \
  -e INFINITETALK_CKPT_DIR=/workspace/weights/Wan2.1-I2V-14B-480P \
  -e INFINITETALK_WAV2VEC_DIR=/workspace/weights/chinese-wav2vec2-base \
  -e INFINITETALK_MODEL_PATH=/workspace/weights/InfiniteTalk/single/infinitetalk.safetensors \
  -e INFINITETALK_KOKORO_DIR=/workspace/weights/Kokoro-82M \
  -e INFINITETALK_DATA_ROOT=/workspace/runtime_data \
  -e INFINITETALK_AUTO_DOWNLOAD_MODELS=true \
  -e INFINITETALK_AUTO_DOWNLOAD_KOKORO=false \
  -e HF_TOKEN=hf_xxx_if_needed \
  infinitetalk-api:latest
```

Notes:

- The container now runs a bootstrap step before `uvicorn`.
- Missing core models are downloaded automatically when `INFINITETALK_AUTO_DOWNLOAD_MODELS=true`.
- Kokoro TTS weights are optional and controlled by `INFINITETALK_AUTO_DOWNLOAD_KOKORO`.
- On Runpod, mounting your network volume at `/workspace` is the simplest layout.

Or:

```bash
docker compose -f docker-compose.gpu.yml up --build
```
