import csv
import json
import tarfile

import torch

from extract_latent.cli import _default_partition_output_names, _select_partition_records
from extract_latent.metadata import load_metadata, make_sample_key
from extract_latent.storage import PthShardWriter, build_sample_payload


def test_load_metadata_jsonl_resolves_paths_and_marks_missing_first_frame(tmp_path):
    video_path = tmp_path / "clips" / "demo clip.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"not-a-real-video")
    metadata_path = tmp_path / "metadata.jsonl"
    metadata_path.write_text(
        json.dumps({"video_path": "clips/demo clip.mp4", "prompt": "a small test video"}) + "\n",
        encoding="utf-8",
    )

    records = load_metadata(metadata_path)

    assert len(records) == 1
    assert records[0].video_path == video_path
    assert records[0].prompt == "a small test video"
    assert records[0].first_frame_path is None
    assert records[0].needs_first_frame_from_video is True


def test_load_metadata_csv_accepts_first_frame_aliases(tmp_path):
    video_path = tmp_path / "video.mp4"
    frame_path = tmp_path / "frame.png"
    video_path.write_bytes(b"video")
    frame_path.write_bytes(b"png")
    metadata_path = tmp_path / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "mp4", "caption", "first_frame"])
        writer.writeheader()
        writer.writerow({"id": "sample-1", "mp4": "video.mp4", "caption": "caption text", "first_frame": "frame.png"})

    records = load_metadata(metadata_path)

    assert records[0].key == "sample-1"
    assert records[0].video_path == video_path
    assert records[0].prompt == "caption text"
    assert records[0].first_frame_path == frame_path
    assert records[0].needs_first_frame_from_video is False


def test_make_sample_key_uses_metadata_key_or_sanitized_video_stem(tmp_path):
    explicit = load_metadata(
        _write_jsonl(tmp_path / "explicit.jsonl", [{"key": "scene 01/A", "video": "clip.mp4", "text": "x"}])
    )[0]
    fallback = load_metadata(
        _write_jsonl(tmp_path / "fallback.jsonl", [{"video": "My Clip 01.mp4", "text": "x"}])
    )[0]

    assert make_sample_key(explicit, 0) == "scene_01_A"
    assert make_sample_key(fallback, 7) == "000007_My_Clip_01"


def test_pth_shard_writer_writes_fastgen_latent_layout(tmp_path):
    latent = torch.randn(48, 21, 4, 6, dtype=torch.bfloat16)
    text = torch.randn(512, 4096, dtype=torch.bfloat16)
    first_frame = torch.randn(48, 1, 4, 6, dtype=torch.bfloat16)
    payload = build_sample_payload(
        key="sample_000001",
        latent=latent,
        text_embedding=text,
        first_frame_cond=first_frame,
        metadata={"prompt": "test"},
    )

    with PthShardWriter(tmp_path, maxcount=1) as writer:
        writer.write(payload)

    tar_path = tmp_path / "00000.tar"
    assert tar_path.exists()
    with tarfile.open(tar_path) as tar:
        names = sorted(tar.getnames())
        assert names == [
            "sample_000001.first_frame_cond.pth",
            "sample_000001.json",
            "sample_000001.latent.pth",
            "sample_000001.txt_emb.pth",
        ]

    shard_counts = json.loads((tmp_path / "shard_count.json").read_text(encoding="utf-8"))
    assert shard_counts == {"00000.tar": 1}


def test_pth_shard_writer_supports_start_index_and_count_filename(tmp_path):
    latent = torch.randn(48, 21, 4, 6, dtype=torch.bfloat16)
    text = torch.randn(512, 4096, dtype=torch.bfloat16)

    with PthShardWriter(tmp_path, maxcount=1, start_index=12, count_filename="shard_count_part00003.json") as writer:
        writer.write(build_sample_payload(key="sample_a", latent=latent, text_embedding=text))
        writer.write(build_sample_payload(key="sample_b", latent=latent, text_embedding=text))

    assert (tmp_path / "00012.tar").exists()
    assert (tmp_path / "00013.tar").exists()
    shard_counts = json.loads((tmp_path / "shard_count_part00003.json").read_text(encoding="utf-8"))
    assert shard_counts == {"00012.tar": 1, "00013.tar": 1}


def test_select_partition_records_preserves_original_indices():
    records = list("abcdefg")

    assert _select_partition_records(records, num_partitions=3, partition_index=0) == [(0, "a"), (3, "d"), (6, "g")]
    assert _select_partition_records(records, num_partitions=3, partition_index=1) == [(1, "b"), (4, "e")]
    assert _select_partition_records(records, num_partitions=3, partition_index=2) == [(2, "c"), (5, "f")]


def test_default_partition_output_names_are_collision_safe_for_parallel_runs():
    assert _default_partition_output_names(num_partitions=1, partition_index=0) == (
        "extract_report.json",
        "shard_count.json",
        0,
        True,
    )
    assert _default_partition_output_names(num_partitions=8, partition_index=3) == (
        "extract_report_part00003.json",
        "shard_count_part00003.json",
        30000,
        False,
    )


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path
