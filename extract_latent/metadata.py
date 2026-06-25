import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


VIDEO_KEYS = ("video_path", "mp4_path", "mp4", "video", "path", "file", "filename")
PROMPT_KEYS = ("prompt", "text", "caption", "txt")
FIRST_FRAME_KEYS = (
    "first_frame_path",
    "first_frame",
    "image_path",
    "image",
    "start_image",
    "conditioning_image",
    "condition_image",
)
KEY_KEYS = ("key", "id", "sample_id", "__key__")


@dataclass(frozen=True)
class MetadataRecord:
    key: str | None
    video_path: Path
    prompt: str
    first_frame_path: Path | None
    raw: dict[str, Any]

    @property
    def needs_first_frame_from_video(self) -> bool:
        return self.first_frame_path is None


def load_metadata(path: str | Path, base_dir: str | Path | None = None) -> list[MetadataRecord]:
    """Load JSONL, JSON, or CSV metadata into normalized records.

    Supported aliases:
    - video: video_path, mp4_path, mp4, video, path, file, filename
    - prompt: prompt, text, caption, txt
    - first frame: first_frame_path, first_frame, image_path, image, start_image,
      conditioning_image, condition_image
    - key: key, id, sample_id, __key__
    """
    metadata_path = Path(path)
    root = Path(base_dir) if base_dir is not None else metadata_path.parent
    rows = _read_rows(metadata_path)
    records = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Metadata row {index} must be an object, got {type(row).__name__}")
        video_value = _first_value(row, VIDEO_KEYS)
        prompt_value = _first_value(row, PROMPT_KEYS)
        if _is_empty(video_value):
            raise ValueError(f"Metadata row {index} is missing a video path key: {VIDEO_KEYS}")
        if _is_empty(prompt_value):
            raise ValueError(f"Metadata row {index} is missing a prompt key: {PROMPT_KEYS}")

        first_frame_value = _first_value(row, FIRST_FRAME_KEYS)
        key_value = _first_value(row, KEY_KEYS)
        records.append(
            MetadataRecord(
                key=None if _is_empty(key_value) else str(key_value),
                video_path=_resolve_path(video_value, root),
                prompt=str(prompt_value),
                first_frame_path=None if _is_empty(first_frame_value) else _resolve_path(first_frame_value, root),
                raw=dict(row),
            )
        )
    return records


def make_sample_key(record: MetadataRecord, index: int) -> str:
    """Build a WebDataset-safe key for a metadata record."""
    if record.key:
        source = record.key
    else:
        source = f"{index:06d}_{record.video_path.stem}"
    key = re.sub(r"[^A-Za-z0-9._-]+", "_", str(source)).strip("._-")
    return key or f"sample_{index:06d}"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            value = json.load(f)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            for key in ("samples", "data", "records", "items"):
                if isinstance(value.get(key), list):
                    return value[key]
            return [value]
        raise ValueError(f"Unsupported JSON metadata root type: {type(value).__name__}")

    if suffix == ".csv":
        with path.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    raise ValueError(f"Unsupported metadata format for {path}; use .jsonl, .json, or .csv")


def _resolve_path(value: Any, root: Path) -> Path:
    text = os.path.expanduser(str(value))
    path = Path(text)
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=False)


def _first_value(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")
