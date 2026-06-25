# 内网 H100 Wan2.2-5B DMD2 Smoke 训练 README

目标：在内网 H100 服务器上，用 Wan2.2-TI2V-5B-Diffusers 和全量 smoke 数据跑 FastGen DMD2 训练，并保留完整 optimizer checkpoint。

不要直接使用 AutoDL smoke config。AutoDL 版本为了避开 50GB 小盘，设置了 `save_optimizer=False`，且路径都指向 `/root/autodl-tmp`。

## 1. 先改这些本地路径

在 H100 服务器上先设置环境变量。下面所有 `TODO(本地路径)` 都需要按内网实际挂载盘修改。

```bash
export FASTGEN_ROOT=/mnt/TODO_LOCAL_CODE/FastGen
export DATA_ROOT=/mnt/TODO_LOCAL_DATA/fastgen_data
export WAN22_MODEL_DIR=/mnt/TODO_LOCAL_MODEL/Wan2.2-TI2V-5B-Diffusers
export HF_HOME=/mnt/TODO_LOCAL_CACHE/huggingface
export FASTGEN_OUTPUT_ROOT=/mnt/TODO_LARGE_OUTPUT/fastgen_outputs

export SMOKE_TAR_DIR="$DATA_ROOT/vipe_fliter_complete"
export RAW_ROOT="$DATA_ROOT/vipe_short_raw"
export METADATA_PATH="$DATA_ROOT/vipe_short/smoke_metadata_full.jsonl"
export LATENT_DIR="$DATA_ROOT/vipe_short_latents_full"
```

要求：
- `FASTGEN_OUTPUT_ROOT` 必须是大盘。保留 optimizer 时，一个完整 5B DMD2 FSDP checkpoint 可能是 60GB 级别，建议至少预留 300GB。
- `DATA_ROOT` 和 `FASTGEN_OUTPUT_ROOT` 不要放在系统盘。
- 训练前确认 `df -h "$DATA_ROOT" "$FASTGEN_OUTPUT_ROOT"`。

## 2. 环境安装

```bash
cd "$FASTGEN_ROOT"

conda create -n fastgen python=3.12 -y
conda activate fastgen

python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

如果内网不能直接访问 Hugging Face，把模型和数据提前同步到上面配置的本地路径即可。

## 3. 准备模型和全量 smoke tar

已有模型时跳过下载，只确认目录存在：

```bash
test -d "$WAN22_MODEL_DIR/transformer"
test -d "$WAN22_MODEL_DIR/vae"
test -d "$WAN22_MODEL_DIR/text_encoder"
```

需要下载时：

```bash
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --local-dir "$WAN22_MODEL_DIR"
```

下载全量 smoke tar。这里按 `short/*.tar` 作为全量 smoke 集合；如果你们内网已有镜像，把 `SMOKE_TAR_DIR` 指向已有 tar 目录。

```bash
mkdir -p "$SMOKE_TAR_DIR"
huggingface-cli download csusupergear/vipe_fliter_complete \
  --repo-type dataset \
  --include "short/*.tar" \
  --local-dir "$SMOKE_TAR_DIR"
```

## 4. 解包全量 smoke 数据

```bash
mkdir -p "$RAW_ROOT"

find "$SMOKE_TAR_DIR" -type f -name "*.tar" | sort | while read -r tar_path; do
  shard="$(basename "$tar_path" .tar)"
  out_dir="$RAW_ROOT/$shard"
  mkdir -p "$out_dir"
  tar -xf "$tar_path" -C "$out_dir"
done
```

## 5. 生成全量 metadata

这一步不会限制 `limit=8`，会递归收集所有 mp4。若同名 `.txt` 文件存在，用它作为 prompt；否则使用兜底 prompt。

```bash
mkdir -p "$(dirname "$METADATA_PATH")"

python - <<'PY'
import json
import os
from pathlib import Path

raw_root = Path(os.environ["RAW_ROOT"])
out = Path(os.environ["METADATA_PATH"])

rows = []
for idx, video in enumerate(sorted(raw_root.rglob("*.mp4"))):
    prompt_path = video.with_suffix(".txt")
    prompt = prompt_path.read_text(encoding="utf-8").strip() if prompt_path.exists() else "a short video"
    rows.append({
        "key": f"smoke_full_{idx:08d}_{video.stem}",
        "video_path": str(video),
        "prompt": prompt,
    })

out.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
print(f"metadata rows: {len(rows)} -> {out}")
PY

wc -l "$METADATA_PATH"
```

## 6. 抽取 latent WebDataset

```bash
cd "$FASTGEN_ROOT"
mkdir -p "$LATENT_DIR"

python -m extract_latent \
  --metadata "$METADATA_PATH" \
  --output-dir "$LATENT_DIR" \
  --task ti2v \
  --model-id-or-local-path "$WAN22_MODEL_DIR" \
  --sequence-length 81 \
  --width 1280 \
  --height 704 \
  --save-dtype bfloat16 \
  --maxcount 1000 \
  --decode-max-samples 0 \
  --skip-errors
```

检查输出：

```bash
test -f "$LATENT_DIR/neg_prompt_emb.npy"
test -f "$LATENT_DIR/extract_report.json"
find "$LATENT_DIR" -maxdepth 1 -name "*.tar" | wc -l
```

### 6.1 8 卡并行抽取 latent

单条 `python -m extract_latent` 默认只会加载一个模型实例并使用一张 GPU。8 卡 H100 上建议启动 8 个进程，每个进程固定一张卡，并用 `--num-partitions/--partition-index` 处理 metadata 的一个互斥子集。

下面命令会把所有 shard 写到同一个 `$LATENT_DIR`，文件名自动错开：

- partition 0: `00000.tar`, `00001.tar`, ...
- partition 1: `10000.tar`, `10001.tar`, ...
- partition 2: `20000.tar`, `20001.tar`, ...

`neg_prompt_emb.npy` 只由 partition 0 写一次；训练仍然使用同一个 `LATENT_DIR`。

```bash
cd "$FASTGEN_ROOT"
rm -rf "$LATENT_DIR"
mkdir -p "$LATENT_DIR/logs"

NUM_GPUS=8
for GPU in $(seq 0 $((NUM_GPUS - 1))); do
  CUDA_VISIBLE_DEVICES="$GPU" python -m extract_latent \
    --metadata "$METADATA_PATH" \
    --output-dir "$LATENT_DIR" \
    --task ti2v \
    --model-id-or-local-path "$WAN22_MODEL_DIR" \
    --sequence-length 81 \
    --width 1280 \
    --height 704 \
    --save-dtype bfloat16 \
    --maxcount 1000 \
    --decode-max-samples 0 \
    --skip-errors \
    --device cuda \
    --num-partitions "$NUM_GPUS" \
    --partition-index "$GPU" \
    > "$LATENT_DIR/logs/extract_part${GPU}.log" 2>&1 &
done
wait
```

合并每个进程写出的 shard count，便于后续 deterministic/resume 场景使用：

```bash
python - <<'PY'
import glob
import json
import os
from pathlib import Path

latent_dir = Path(os.environ["LATENT_DIR"])
merged = {}
for path in sorted(glob.glob(str(latent_dir / "shard_count_part*.json"))):
    merged.update(json.loads(Path(path).read_text(encoding="utf-8")))
(latent_dir / "shard_count.json").write_text(
    json.dumps(merged, indent=2, sort_keys=True),
    encoding="utf-8",
)
print(f"merged shards: {len(merged)} -> {latent_dir / 'shard_count.json'}")
PY
```

检查并行抽取结果：

```bash
test -f "$LATENT_DIR/neg_prompt_emb.npy"
find "$LATENT_DIR" -maxdepth 1 -name "*.tar" | sort | head
find "$LATENT_DIR" -maxdepth 1 -name "*.tar" | wc -l
python - <<'PY'
import glob
import json
import os
from pathlib import Path

total_samples = 0
total_errors = 0
for path in sorted(glob.glob(os.path.join(os.environ["LATENT_DIR"], "extract_report_part*.json"))):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    total_samples += len(report["samples"])
    total_errors += len(report["errors"])
print({"samples": total_samples, "errors": total_errors})
PY
```

## 7. 创建内网 H100 训练 config

创建新文件，不要覆盖 AutoDL config：

```bash
mkdir -p "$FASTGEN_ROOT/fastgen/configs/experiments/local"
touch "$FASTGEN_ROOT/fastgen/configs/experiments/local/__init__.py"
```

写入 `fastgen/configs/experiments/local/wan22_i2v_dmd2_smoke_latent_h100_full.py`。下面两行 `TODO(本地路径)` 必须改成你的服务器真实路径；不要保留 `/mnt/TODO...`。

```python
from copy import deepcopy

from fastgen.configs.data import VideoLatentLoaderConfig
from fastgen.configs.experiments.WanI2V.config_dmd2_wan22_5b import create_config as create_base_config


LATENT_DIR = "/mnt/TODO_LOCAL_DATA/fastgen_data/vipe_short_latents_full"  # TODO(本地路径)
MODEL_DIR = "/mnt/TODO_LOCAL_MODEL/Wan2.2-TI2V-5B-Diffusers"  # TODO(本地路径)


def create_config():
    config = create_base_config()

    config.model.net.model_id_or_local_path = MODEL_DIR
    config.model.enable_preprocessors = False

    config.dataloader_train = deepcopy(VideoLatentLoaderConfig)
    config.dataloader_train.datatags = [f"WDS:{LATENT_DIR}"]
    config.dataloader_train.batch_size = 1
    config.dataloader_train.num_workers = 4
    config.dataloader_train.key_map = {
        "real": "latent.pth",
        "condition": "txt_emb.pth",
        "first_frame_cond": "first_frame_cond.pth",
    }
    config.dataloader_train.files_map = {
        "neg_condition": f"{LATENT_DIR}/neg_prompt_emb.npy",
    }

    config.trainer.fsdp = True
    config.trainer.ddp = False
    config.trainer.resume = False
    config.trainer.logging_iter = 10
    config.trainer.validation_iter = 1_000_000

    # TODO(训练长度): 按 smoke 样本数和想跑的 epoch 数调整。
    # 训练循环实际执行 range(1, max_iter)，所以 max_iter=501 表示 500 个 step。
    config.trainer.max_iter = 501
    config.trainer.save_ckpt_iter = 100

    # 保留完整 checkpoint，正式内网 H100 训练不要关闭 optimizer。
    config.trainer.checkpointer.save_optimizer = True
    config.trainer.checkpointer.save_scheduler = True
    config.trainer.checkpointer.save_grad_scaler = True
    config.trainer.checkpointer.save_callbacks = True

    # smoke 训练建议先关 W&B media callback，避免 iteration 1 强制 VAE sample logging。
    config.trainer.callbacks.pop("wandb", None)

    config.log_config.group = "intranet_h100_smoke"
    config.log_config.name = "wan22_i2v_dmd2_latent_full"
    config.log_config.wandb_mode = "disabled"
    return config
```

## 8. Dry run

```bash
cd "$FASTGEN_ROOT"
export FASTGEN_OUTPUT_ROOT=/mnt/TODO_LARGE_OUTPUT/fastgen_outputs  # TODO(本地路径)

python train.py \
  --config=fastgen/configs/experiments/local/wan22_i2v_dmd2_smoke_latent_h100_full.py \
  --dryrun
```

检查 `config.yaml` 中：
- `model.net.model_id_or_local_path` 是内网模型路径。
- `dataloader_train.datatags` 是 `WDS:$LATENT_DIR`。
- `trainer.checkpointer.save_optimizer` 是 `True`。
- `trainer.checkpointer.save_dir` 落在大盘 `$FASTGEN_OUTPUT_ROOT` 下。

## 9. 启动训练

按 H100 数量修改 `NPROC_PER_NODE`。

```bash
cd "$FASTGEN_ROOT"
export FASTGEN_OUTPUT_ROOT=/mnt/TODO_LARGE_OUTPUT/fastgen_outputs  # TODO(本地路径)
export WANDB_MODE=disabled

NPROC_PER_NODE=8  # TODO(本地机器): 改成实际 H100 数量

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  train.py \
  --config=fastgen/configs/experiments/local/wan22_i2v_dmd2_smoke_latent_h100_full.py \
  2>&1 | tee "$FASTGEN_OUTPUT_ROOT/train_wan22_dmd2_smoke_h100_full.log"
```

## 10. 成功检查

```bash
OUT="$FASTGEN_OUTPUT_ROOT/fastgen/intranet_h100_smoke/wan22_i2v_dmd2_latent_full"

grep -E "Training complete|Model saved|Training finished" "$FASTGEN_OUTPUT_ROOT/train_wan22_dmd2_smoke_h100_full.log"
tail -n 20 "$OUT/metrics.jsonl"
find "$OUT/checkpoints" -maxdepth 1 -mindepth 1 | sort
du -sh "$OUT/checkpoints"
```

保留 optimizer 时，checkpoint 下应能看到模型分片和 optimizer 分片，例如：

```text
0000100.net_model
0000100.net_optim
0000100.fake_score_model
0000100.fake_score_optim
0000100.discriminator_model
0000100.discriminator_optim
0000100.pth
```

如果写 checkpoint 时报 `unexpected pos` 或 `No space left on device`，优先检查 `FASTGEN_OUTPUT_ROOT` 是否真正在大盘上，而不是降低 checkpoint 内容。
