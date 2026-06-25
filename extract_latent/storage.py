import io
import json
import tarfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def build_sample_payload(
    *,
    key: str,
    latent: torch.Tensor,
    text_embedding: torch.Tensor,
    first_frame_cond: torch.Tensor | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one WebDataset sample using FastGen's latent file conventions."""
    payload: dict[str, Any] = {
        "__key__": key,
        "latent.pth": _cpu_tensor(latent),
        "txt_emb.pth": _cpu_tensor(text_embedding),
    }
    if first_frame_cond is not None:
        payload["first_frame_cond.pth"] = _cpu_tensor(first_frame_cond)
    if metadata is not None:
        payload["json"] = dict(metadata)
    return payload


def save_shared_embedding(path: str | Path, embedding: torch.Tensor) -> None:
    """Save a shared embedding for WDSLoader.files_map.

    NumPy has no stable bfloat16 dtype, so shared embeddings are written as
    float32. Trainer.preprocess_data moves them to model.precision afterward.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = embedding.detach().float().cpu().numpy()
    np.save(path, array)


class PthShardWriter:
    """Minimal WebDataset tar writer for torch-saved latent samples."""

    def __init__(
        self,
        output_dir: str | Path,
        maxcount: int = 1000,
        pattern: str = "%05d.tar",
        start_index: int = 0,
        count_filename: str = "shard_count.json",
    ):
        if maxcount <= 0:
            raise ValueError("maxcount must be positive")
        if start_index < 0:
            raise ValueError("start_index must be non-negative")
        self.output_dir = Path(output_dir)
        self.maxcount = maxcount
        self.pattern = pattern
        self.count_filename = count_filename
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._tar: tarfile.TarFile | None = None
        self._shard_index = start_index - 1
        self._count_in_shard = 0
        self._shard_counts: dict[str, int] = {}

    def __enter__(self) -> "PthShardWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write(self, sample: Mapping[str, Any]) -> None:
        if "__key__" not in sample:
            raise ValueError("sample must contain '__key__'")
        if self._tar is None or self._count_in_shard >= self.maxcount:
            self._open_next_shard()

        assert self._tar is not None
        sample_key = str(sample["__key__"])
        for extension, value in sample.items():
            if extension == "__key__":
                continue
            member_name = f"{sample_key}.{extension}"
            data = _serialize_value(extension, value)
            info = tarfile.TarInfo(member_name)
            info.size = len(data)
            self._tar.addfile(info, io.BytesIO(data))

        self._count_in_shard += 1
        self._shard_counts[self._current_shard_name()] = self._count_in_shard

    def close(self) -> None:
        if self._tar is not None:
            self._tar.close()
            self._tar = None
        count_path = self.output_dir / self.count_filename
        count_path.write_text(json.dumps(self._shard_counts, indent=2, sort_keys=True), encoding="utf-8")

    def _open_next_shard(self) -> None:
        if self._tar is not None:
            self._tar.close()
        self._shard_index += 1
        self._count_in_shard = 0
        shard_path = self.output_dir / self._current_shard_name()
        self._tar = tarfile.open(shard_path, "w")

    def _current_shard_name(self) -> str:
        return self.pattern % self._shard_index


def _cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().contiguous()


def _serialize_value(extension: str, value: Any) -> bytes:
    if extension.endswith(".pth"):
        buffer = io.BytesIO()
        torch.save(value, buffer)
        return buffer.getvalue()
    if extension == "json" or extension.endswith(".json"):
        return json.dumps(value, ensure_ascii=True).encode("utf-8")
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    raise TypeError(f"Unsupported WebDataset value for extension {extension!r}: {type(value).__name__}")
