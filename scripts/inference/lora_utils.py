# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any


LORA_ADAPTER_NAME = "default"
_LORA_FILE_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth"}


def _resolve_lora_source(lora_path: str) -> tuple[str, dict[str, str]]:
    path = Path(lora_path).expanduser()
    local_path = path if path.is_absolute() else Path.cwd() / path

    if local_path.is_file():
        return str(local_path.parent), {"weight_name": local_path.name}
    if local_path.is_dir():
        return str(local_path), {}
    if path.suffix in _LORA_FILE_SUFFIXES:
        raise FileNotFoundError(f"LoRA file not found: {local_path}")

    return lora_path, {}


def _has_adapter(transformer: Any, adapter_name: str) -> bool:
    return adapter_name in getattr(transformer, "peft_config", {})


def load_lora_adapter(module: Any, args: Any, ctx: dict[str, Any]) -> bool:
    lora_path = getattr(args, "lora_path", None)
    if module is None or lora_path is None:
        return False

    transformer = getattr(module, "transformer", None)
    if transformer is None:
        raise ValueError(f"Cannot load LoRA because {type(module).__name__} has no transformer attribute")

    source, source_kwargs = _resolve_lora_source(lora_path)
    load_kwargs = {"adapter_name": LORA_ADAPTER_NAME, "prefix": "transformer", **source_kwargs}
    transformer.load_lora_adapter(source, **load_kwargs)

    if not _has_adapter(transformer, LORA_ADAPTER_NAME):
        load_kwargs["prefix"] = None
        transformer.load_lora_adapter(source, **load_kwargs)

    if not _has_adapter(transformer, LORA_ADAPTER_NAME):
        raise RuntimeError(f"No LoRA adapter was loaded from {lora_path}")

    transformer.set_adapters(LORA_ADAPTER_NAME, weights=args.lora_scale)
    module.to(**ctx).eval().requires_grad_(False)
    return True
