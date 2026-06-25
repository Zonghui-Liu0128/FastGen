# Wan2.2 Latent Extraction

This folder provides an offline equivalent of `VideoWDSLoader + Trainer.preprocess_data()` for Wan2.2-TI2V-5B.

It reads metadata rows containing an mp4 path and text prompt, encodes video frames with `WanVideoEncoder`, encodes prompts with `WanTextEncoder`, optionally creates `first_frame_cond` for TI2V/I2V, and writes FastGen-compatible WebDataset shards.

## Metadata

Supported formats: `.jsonl`, `.json`, `.csv`.

Supported aliases:

- Video: `video_path`, `mp4_path`, `mp4`, `video`, `path`, `file`, `filename`
- Prompt: `prompt`, `text`, `caption`, `txt`
- First frame: `first_frame_path`, `first_frame`, `image_path`, `image`, `start_image`, `conditioning_image`, `condition_image`
- Key: `key`, `id`, `sample_id`, `__key__`

If `--task ti2v` or `--task i2v` is used and no first-frame path is present, the tool uses the transformed first frame of the mp4 as `first_frame_cond`.

Example JSONL:

```jsonl
{"key":"sample_000001","video_path":"videos/a.mp4","prompt":"a person walking in a city"}
{"key":"sample_000002","mp4":"videos/b.mp4","caption":"a dog running","first_frame":"frames/b.png"}
```

## Run

```bash
python -m extract_latent \
  --metadata /path/to/metadata.jsonl \
  --output-dir /path/to/wan22_latents \
  --task ti2v \
  --model-id-or-local-path Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --sequence-length 81 \
  --width 1280 \
  --height 704 \
  --save-dtype bfloat16 \
  --decode-max-samples 4
```

Outputs:

- `00000.tar`, `00001.tar`, ... with per-sample `latent.pth`, `txt_emb.pth`, and for TI2V/I2V `first_frame_cond.pth`
- `neg_prompt_emb.npy` for `WDSLoader.files_map`
- `shard_count.json` for deterministic loaders
- `extract_report.json` with shapes, dtypes, first-frame source, and decoded preview paths
- `decoded_previews/*.mp4` for a few decoded latents

Expected Wan2.2 5B default tensor layout:

- `latent.pth`: `[48, 21, 44, 80]`
- `txt_emb.pth`: `[512, 4096]`
- `first_frame_cond.pth`: `[48, 1, 44, 80]`
- `neg_prompt_emb.npy`: `[512, 4096]`, loaded once as `neg_condition`

## Multi-GPU Extraction

`extract_latent` is single-process by default. On multi-GPU machines, launch one process per GPU and partition the same metadata file:

```bash
NUM_GPUS=8
for GPU in $(seq 0 $((NUM_GPUS - 1))); do
  CUDA_VISIBLE_DEVICES="$GPU" python -m extract_latent \
    --metadata /path/to/metadata.jsonl \
    --output-dir /path/to/wan22_latents \
    --task ti2v \
    --model-id-or-local-path /path/to/Wan2.2-TI2V-5B-Diffusers \
    --sequence-length 81 \
    --width 1280 \
    --height 704 \
    --save-dtype bfloat16 \
    --decode-max-samples 0 \
    --device cuda \
    --num-partitions "$NUM_GPUS" \
    --partition-index "$GPU" \
    > "/path/to/wan22_latents/extract_part${GPU}.log" 2>&1 &
done
wait
```

When `--num-partitions > 1`, default outputs are collision-safe:

- shard indices start at `partition_index * 10000`
- reports are written as `extract_report_partXXXXX.json`
- shard counts are written as `shard_count_partXXXXX.json`
- only partition 0 writes the shared `neg_prompt_emb.npy`

## Training Loader

T2V latent training can use `VideoLatentLoaderConfig` directly:

```python
from fastgen.configs.data import VideoLatentLoaderConfig

config.dataloader_train = VideoLatentLoaderConfig
config.dataloader_train.datatags = ["WDS:/path/to/wan22_latents"]
config.dataloader_train.files_map = {"neg_condition": "/path/to/wan22_latents/neg_prompt_emb.npy"}
```

TI2V/I2V latent training must add the first-frame condition key:

```python
from fastgen.configs.data import VideoLatentLoaderConfig

config.dataloader_train = VideoLatentLoaderConfig
config.dataloader_train.datatags = ["WDS:/path/to/wan22_latents"]
config.dataloader_train.key_map = {
    "real": "latent.pth",
    "condition": "txt_emb.pth",
    "first_frame_cond": "first_frame_cond.pth",
}
config.dataloader_train.files_map = {"neg_condition": "/path/to/wan22_latents/neg_prompt_emb.npy"}
```

`Trainer.preprocess_data()` will move these tensors to `model.device` and `model.precision`. It will not re-encode `first_frame_cond` when the batch already contains it.
