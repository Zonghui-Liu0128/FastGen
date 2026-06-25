import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from extract_latent.metadata import load_metadata, make_sample_key
from extract_latent.storage import PthShardWriter, build_sample_payload, save_shared_embedding
from extract_latent.video import (
    first_frame_from_transformed_video,
    load_first_frame_image,
    read_video_frames,
    transform_video_frames,
    write_video_preview,
)
from extract_latent.wan import WanLatentExtractor, parse_torch_dtype


NEG_PROMPT_WAN = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
    "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
    "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
    "fused fingers, still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)


def _select_partition_records(records: list[Any], *, num_partitions: int, partition_index: int) -> list[tuple[int, Any]]:
    if num_partitions <= 0:
        raise ValueError("num_partitions must be positive")
    if partition_index < 0 or partition_index >= num_partitions:
        raise ValueError(f"partition_index must be in [0, {num_partitions}), got {partition_index}")
    return [
        (index, record)
        for index, record in enumerate(records)
        if index % num_partitions == partition_index
    ]


def _default_partition_output_names(
    *, num_partitions: int, partition_index: int
) -> tuple[str, str, int, bool]:
    if num_partitions <= 0:
        raise ValueError("num_partitions must be positive")
    if partition_index < 0 or partition_index >= num_partitions:
        raise ValueError(f"partition_index must be in [0, {num_partitions}), got {partition_index}")
    if num_partitions == 1:
        return "extract_report.json", "shard_count.json", 0, True
    return (
        f"extract_report_part{partition_index:05d}.json",
        f"shard_count_part{partition_index:05d}.json",
        partition_index * 10_000,
        partition_index == 0,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    img_size = (args.width, args.height)
    save_dtype = parse_torch_dtype(args.save_dtype)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_metadata(args.metadata, base_dir=args.base_dir)
    partition_records = _select_partition_records(
        records,
        num_partitions=args.num_partitions,
        partition_index=args.partition_index,
    )
    default_report_name, default_shard_count_name, default_shard_start_index, default_write_neg_embedding = (
        _default_partition_output_names(
            num_partitions=args.num_partitions,
            partition_index=args.partition_index,
        )
    )
    report_name = args.report_name or default_report_name
    shard_count_name = args.shard_count_name or default_shard_count_name
    shard_start_index = args.shard_start_index
    if shard_start_index is None:
        shard_start_index = default_shard_start_index
    write_neg_embedding = args.write_negative_prompt_embedding
    if write_neg_embedding is None:
        write_neg_embedding = default_write_neg_embedding

    extractor = WanLatentExtractor(
        model_id_or_local_path=args.model_id_or_local_path,
        device=device,
        save_dtype=save_dtype,
        text_precision=parse_torch_dtype(args.text_precision),
    )

    if write_neg_embedding:
        neg_embedding = extractor.encode_text(args.negative_prompt)
        save_shared_embedding(output_dir / "neg_prompt_emb.npy", neg_embedding)

    expected_latent_shape = _expected_wan22_latent_shape(args.sequence_length, img_size)
    report: dict[str, Any] = {
        "metadata": str(Path(args.metadata).resolve(strict=False)),
        "output_dir": str(output_dir.resolve(strict=False)),
        "task": args.task,
        "model_id_or_local_path": args.model_id_or_local_path,
        "sequence_length": args.sequence_length,
        "img_size": {"width": args.width, "height": args.height},
        "num_metadata_records": len(records),
        "num_selected_records": len(partition_records),
        "partition": {
            "num_partitions": args.num_partitions,
            "partition_index": args.partition_index,
            "shard_start_index": shard_start_index,
            "report_name": report_name,
            "shard_count_name": shard_count_name,
            "write_negative_prompt_embedding": write_neg_embedding,
        },
        "save_dtype": str(save_dtype).replace("torch.", ""),
        "expected_latent_shape": list(expected_latent_shape),
        "samples": [],
        "errors": [],
    }

    preview_dir = Path(args.decode_preview_dir) if args.decode_preview_dir else output_dir / "decoded_previews"
    preview_limit = args.decode_max_samples

    with PthShardWriter(
        output_dir,
        maxcount=args.maxcount,
        start_index=shard_start_index,
        count_filename=shard_count_name,
    ) as writer:
        for index, record in tqdm(partition_records, desc="extract_latent"):
            key = make_sample_key(record, index)
            try:
                frames = read_video_frames(record.video_path, args.sequence_length)
                video, cropping_params = transform_video_frames(
                    frames,
                    sequence_length=args.sequence_length,
                    img_size=img_size,
                )
                first_frame = None
                first_frame_source = None
                if args.task in {"ti2v", "i2v"}:
                    if record.first_frame_path is None:
                        first_frame = first_frame_from_transformed_video(video)
                        first_frame_source = "video_first_frame"
                    else:
                        first_frame = load_first_frame_image(record.first_frame_path, img_size=img_size)
                        first_frame_source = str(record.first_frame_path)

                encoded = extractor.extract(
                    video=video,
                    prompt=record.prompt,
                    first_frame=first_frame,
                    video_vae_mode=args.video_vae_mode,
                )
                _validate_encoded_sample(
                    encoded,
                    expected_latent_shape=expected_latent_shape,
                    require_first_frame=args.task in {"ti2v", "i2v"},
                    strict=args.strict_shape,
                )

                payload = build_sample_payload(
                    key=key,
                    latent=encoded.latent,
                    text_embedding=encoded.text_embedding,
                    first_frame_cond=encoded.first_frame_cond,
                    metadata={
                        **record.raw,
                        "key": key,
                        "prompt": record.prompt,
                        "video_path": str(record.video_path),
                        "first_frame_source": first_frame_source,
                        "cropping_params": cropping_params,
                    },
                )
                writer.write(payload)

                sample_report: dict[str, Any] = {
                    "key": key,
                    "video_path": str(record.video_path),
                    "first_frame_source": first_frame_source,
                    "latent_shape": list(encoded.latent.shape),
                    "text_embedding_shape": list(encoded.text_embedding.shape),
                    "latent_dtype": str(encoded.latent.dtype).replace("torch.", ""),
                    "text_embedding_dtype": str(encoded.text_embedding.dtype).replace("torch.", ""),
                }
                if encoded.first_frame_cond is not None:
                    sample_report["first_frame_cond_shape"] = list(encoded.first_frame_cond.shape)
                    sample_report["first_frame_cond_dtype"] = str(encoded.first_frame_cond.dtype).replace("torch.", "")

                if preview_limit > 0 and len([s for s in report["samples"] if "decoded_preview" in s]) < preview_limit:
                    decoded = extractor.decode_latent(encoded.latent)
                    preview_path = preview_dir / f"{key}.mp4"
                    write_video_preview(preview_path, decoded, fps=args.preview_fps)
                    sample_report["decoded_preview"] = str(preview_path.resolve(strict=False))
                    sample_report["decoded_shape"] = list(decoded.shape)

                report["samples"].append(sample_report)
            except Exception as exc:
                error = {"key": key, "video_path": str(record.video_path), "error": repr(exc)}
                report["errors"].append(error)
                if not args.skip_errors:
                    raise

    report_path = output_dir / report_name
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m extract_latent",
        description="Extract FastGen Wan2.2 video latents from metadata.",
    )
    parser.add_argument("--metadata", required=True, help="Path to .jsonl, .json, or .csv metadata.")
    parser.add_argument("--base-dir", default=None, help="Base directory for relative metadata paths.")
    parser.add_argument("--output-dir", required=True, help="Output directory for WebDataset latent shards.")
    parser.add_argument(
        "--task",
        choices=("t2v", "ti2v", "i2v"),
        default="ti2v",
        help="Use t2v to omit first_frame_cond, ti2v/i2v to save it.",
    )
    parser.add_argument(
        "--model-id-or-local-path",
        default="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        help="HF model ID or local diffusers directory containing vae/text_encoder/tokenizer.",
    )
    parser.add_argument("--sequence-length", type=int, default=81)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--maxcount", type=int, default=1000, help="Samples per tar shard.")
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=1,
        help="Split metadata by index modulo this value for parallel extraction.",
    )
    parser.add_argument(
        "--partition-index",
        type=int,
        default=0,
        help="Current metadata partition index in [0, num_partitions).",
    )
    parser.add_argument(
        "--shard-start-index",
        type=int,
        default=None,
        help="First tar shard index for this process. Defaults to partition_index * 10000 when num_partitions > 1.",
    )
    parser.add_argument(
        "--report-name",
        default=None,
        help="Report filename under output-dir. Defaults to per-partition names when num_partitions > 1.",
    )
    parser.add_argument(
        "--shard-count-name",
        default=None,
        help="Shard count filename under output-dir. Defaults to per-partition names when num_partitions > 1.",
    )
    parser.add_argument(
        "--write-negative-prompt-embedding",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write neg_prompt_emb.npy. Defaults to true for single partition or partition 0 only.",
    )
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu. Defaults to cuda when available.")
    parser.add_argument("--save-dtype", default="bfloat16", help="float32, float16, or bfloat16.")
    parser.add_argument("--text-precision", default="float32", help="Precision requested from WanTextEncoder.")
    parser.add_argument(
        "--video-vae-mode",
        choices=("sample", "argmax"),
        default="sample",
        help="Online Trainer uses sample for real video latents; first_frame_cond always uses argmax.",
    )
    parser.add_argument("--negative-prompt", default=NEG_PROMPT_WAN)
    parser.add_argument("--decode-preview-dir", default=None)
    parser.add_argument("--decode-max-samples", type=int, default=4)
    parser.add_argument("--preview-fps", type=int, default=16)
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--strict-shape", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def _expected_wan22_latent_shape(sequence_length: int, img_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = img_size
    if (sequence_length - 1) % 4 != 0:
        raise ValueError("Wan video sequence_length must satisfy sequence_length = 4 * latent_T - 3")
    if width % 16 != 0 or height % 16 != 0:
        raise ValueError("Wan2.2 latent spatial size requires width and height divisible by 16")
    return 48, (sequence_length - 1) // 4 + 1, height // 16, width // 16


def _validate_encoded_sample(
    encoded,
    *,
    expected_latent_shape: tuple[int, int, int, int],
    require_first_frame: bool,
    strict: bool,
) -> None:
    if strict and tuple(encoded.latent.shape) != expected_latent_shape:
        raise ValueError(f"latent shape {tuple(encoded.latent.shape)} != expected {expected_latent_shape}")
    if encoded.text_embedding.ndim != 2 or encoded.text_embedding.shape[0] != 512:
        raise ValueError(f"text embedding should have shape [512, D], got {tuple(encoded.text_embedding.shape)}")
    if require_first_frame:
        if encoded.first_frame_cond is None:
            raise ValueError("first_frame_cond is required for ti2v/i2v")
        expected_first = (expected_latent_shape[0], 1, expected_latent_shape[2], expected_latent_shape[3])
        if strict and tuple(encoded.first_frame_cond.shape) != expected_first:
            raise ValueError(
                f"first_frame_cond shape {tuple(encoded.first_frame_cond.shape)} != expected {expected_first}"
            )


if __name__ == "__main__":
    main()
