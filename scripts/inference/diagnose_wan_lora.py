# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diagnose whether a Wan Diffusers LoRA is really injected and active.

This script is intentionally standalone so it can run in an offline intranet
environment. It compares the current FastGen v0.3 direct loader behavior with
Diffusers' Wan-specific LoRA loader path, then reports adapter layer counts,
non-zero LoRA weights, active adapters, and an optional forward-difference check.

Example:
    PYTHONPATH=$(pwd) python scripts/inference/diagnose_wan_lora.py \
        --model_path /models/Wan2.2-TI2V-5B-Diffusers \
        --lora_path /models/lora/rotate_figure.safetensors \
        --device cuda --dtype bfloat16 --forward_check
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


LORA_ADAPTER_NAME = "diagnostic"
LORA_FILE_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth"}
COMMON_LORA_WEIGHT_NAMES = (
    "pytorch_lora_weights.safetensors",
    "adapter_model.safetensors",
    "lora.safetensors",
    "pytorch_lora_weights.bin",
    "adapter_model.bin",
)


@dataclass
class LayerStats:
    count: int = 0
    nonzero_a: int = 0
    nonzero_b: int = 0
    total_a_abs: float = 0.0
    total_b_abs: float = 0.0


def log_section(title: str) -> None:
    print(f"\n{'=' * 16} {title} {'=' * 16}", flush=True)


def fail(message: str) -> None:
    print(f"[FAIL] {message}", flush=True)


def warn(message: str) -> None:
    print(f"[WARN] {message}", flush=True)


def ok(message: str) -> None:
    print(f"[OK] {message}", flush=True)


def info(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype {name}. Choose from: {sorted(mapping)}")
    return mapping[name]


def resolve_lora_for_direct_loader(lora_path: str, weight_name: str | None) -> tuple[str, dict[str, str]]:
    path = Path(lora_path).expanduser()
    local_path = path if path.is_absolute() else Path.cwd() / path

    if local_path.is_file():
        return str(local_path.parent), {"weight_name": local_path.name}

    if local_path.is_dir():
        if weight_name is not None:
            return str(local_path), {"weight_name": weight_name}
        return str(local_path), {}

    if path.suffix in LORA_FILE_SUFFIXES:
        raise FileNotFoundError(f"LoRA file not found: {local_path}")

    kwargs = {"weight_name": weight_name} if weight_name else {}
    return lora_path, kwargs


def resolve_lora_for_official_loader(lora_path: str, weight_name: str | None) -> tuple[str, dict[str, str]]:
    path = Path(lora_path).expanduser()
    local_path = path if path.is_absolute() else Path.cwd() / path

    if local_path.is_file():
        return str(local_path.parent), {"weight_name": local_path.name}

    if local_path.is_dir():
        if weight_name is not None:
            return str(local_path), {"weight_name": weight_name}
        local_file = resolve_local_lora_file(lora_path, weight_name)
        if local_file is not None:
            return str(local_file.parent), {"weight_name": local_file.name}
        return str(local_path), {}

    kwargs = {"weight_name": weight_name} if weight_name else {}
    return lora_path, kwargs


def resolve_local_lora_file(lora_path: str, weight_name: str | None) -> Path | None:
    path = Path(lora_path).expanduser()
    local_path = path if path.is_absolute() else Path.cwd() / path

    if local_path.is_file():
        return local_path
    if not local_path.is_dir():
        return None

    if weight_name is not None:
        candidate = local_path / weight_name
        return candidate if candidate.is_file() else None

    for name in COMMON_LORA_WEIGHT_NAMES:
        candidate = local_path / name
        if candidate.is_file():
            return candidate

    safetensors = sorted(local_path.glob("*.safetensors"))
    if len(safetensors) == 1:
        return safetensors[0]

    bins = sorted(local_path.glob("*.bin"))
    if len(bins) == 1:
        return bins[0]

    return None


def load_raw_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")

    loaded = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(loaded, dict):
        for key in ("state_dict", "module", "model", "lora", "network"):
            if isinstance(loaded.get(key), dict):
                loaded = loaded[key]
                break
    if not isinstance(loaded, dict):
        raise TypeError(f"Unsupported LoRA checkpoint object type: {type(loaded)}")
    return {k: v for k, v in loaded.items() if isinstance(v, torch.Tensor)}


def detect_key_family(keys: list[str]) -> Counter:
    families: Counter = Counter()
    for key in keys:
        if key.startswith("transformer."):
            families["diffusers_transformer_prefix"] += 1
        elif key.startswith("blocks."):
            families["wan_or_diffsynth_blocks_prefix"] += 1
        elif key.startswith("diffusion_model."):
            families["original_diffusion_model_prefix"] += 1
        elif key.startswith("lora_unet_"):
            families["musubi_lora_unet_prefix"] += 1
        elif key.startswith("transformer_"):
            families["underscore_transformer_prefix"] += 1
        else:
            families["other"] += 1

        if ".lora_A." in key or key.endswith(".lora_A.weight"):
            families["lora_A"] += 1
        if ".lora_B." in key or key.endswith(".lora_B.weight"):
            families["lora_B"] += 1
        if ".lora_down." in key:
            families["lora_down"] += 1
        if ".lora_up." in key:
            families["lora_up"] += 1
        if key.endswith(".alpha") or ".alpha" in key:
            families["alpha"] += 1
    return families


def summarize_raw_lora(lora_path: str, weight_name: str | None, max_keys: int) -> None:
    log_section("Raw LoRA checkpoint")
    local_file = resolve_local_lora_file(lora_path, weight_name)
    if local_file is None:
        warn("Could not find a local LoRA weight file for raw key inspection.")
        warn("If this is a cached Hugging Face repo id, model-load checks may still work.")
        return

    info(f"Inspecting local LoRA file: {local_file}")
    state_dict = load_raw_state_dict(local_file)
    keys = list(state_dict)
    families = detect_key_family(keys)
    print(f"num_tensors: {len(keys)}")
    print("key_family_counts:")
    for key, value in families.most_common():
        print(f"  {key}: {value}")

    print("sample_keys:")
    for key in keys[:max_keys]:
        tensor = state_dict[key]
        abs_sum = tensor.float().abs().sum().item()
        print(f"  {key} shape={tuple(tensor.shape)} abs_sum={abs_sum:.6e}")

    if families["lora_B"] == 0 and families["lora_up"] == 0:
        fail("No lora_B/lora_up tensors found. This does not look like a standard LoRA checkpoint.")
    elif sum(t.float().abs().sum().item() for k, t in state_dict.items() if "lora_B" in k or "lora_up" in k) == 0:
        fail("All lora_B/lora_up tensors have zero absolute sum. This LoRA will not change outputs.")

    if families["wan_or_diffsynth_blocks_prefix"] or families["diffusion_model_prefix"] or families["musubi_lora_unet_prefix"]:
        warn("Checkpoint keys do not look like already-prefixed Diffusers transformer keys.")
        warn("The current v0.3 direct loader may report loaded but miss Wan-specific key conversion.")


def load_transformer(model_path: str, dtype: torch.dtype, device: str, local_files_only: bool):
    from diffusers.models import WanTransformer3DModel

    transformer = WanTransformer3DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=dtype,
        local_files_only=local_files_only,
        cache_dir=os.environ.get("HF_HOME"),
    )
    transformer.eval().to(device=device)
    return transformer


def peft_layers(transformer: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer

        return [(name, module) for name, module in transformer.named_modules() if isinstance(module, BaseTunerLayer)]
    except Exception:
        return [
            (name, module)
            for name, module in transformer.named_modules()
            if hasattr(module, "lora_A") and hasattr(module, "lora_B")
        ]


def active_adapter_names(transformer: torch.nn.Module) -> Any:
    if hasattr(transformer, "active_adapters"):
        try:
            return transformer.active_adapters()
        except TypeError:
            return transformer.active_adapters
    return None


def inspect_loaded_layers(transformer: torch.nn.Module, adapter_name: str, max_layers: int) -> LayerStats:
    layers = peft_layers(transformer)
    stats = LayerStats(count=len(layers))
    print(f"peft_config_keys: {list(getattr(transformer, 'peft_config', {}).keys())}")
    print(f"active_adapters: {active_adapter_names(transformer)}")
    print(f"lora_tuner_layers: {len(layers)}")

    target_counts = Counter()
    for name, _ in layers:
        if ".attn1." in name:
            target_counts["attn1"] += 1
        elif ".attn2." in name:
            target_counts["attn2"] += 1
        elif ".ffn." in name:
            target_counts["ffn"] += 1
        elif "patch_embedding" in name:
            target_counts["patch_embedding"] += 1
        elif "proj_out" in name:
            target_counts["proj_out"] += 1
        else:
            target_counts["other"] += 1
    print(f"target_group_counts: {dict(target_counts)}")

    for name, module in layers[:max_layers]:
        lora_a = getattr(module, "lora_A", {})
        lora_b = getattr(module, "lora_B", {})
        scaling = getattr(module, "scaling", {})
        has_adapter = adapter_name in lora_a and adapter_name in lora_b
        print(f"  layer={name} has_adapter={has_adapter} scaling={scaling.get(adapter_name, None)}")
        if not has_adapter:
            continue
        a_weight = lora_a[adapter_name].weight.detach().float()
        b_weight = lora_b[adapter_name].weight.detach().float()
        a_abs = a_weight.abs().sum().item()
        b_abs = b_weight.abs().sum().item()
        stats.total_a_abs += a_abs
        stats.total_b_abs += b_abs
        stats.nonzero_a += int(a_abs > 0)
        stats.nonzero_b += int(b_abs > 0)
        print(f"    A_shape={tuple(a_weight.shape)} A_abs={a_abs:.6e}")
        print(f"    B_shape={tuple(b_weight.shape)} B_abs={b_abs:.6e}")

    for _, module in layers[max_layers:]:
        lora_a = getattr(module, "lora_A", {})
        lora_b = getattr(module, "lora_B", {})
        if adapter_name not in lora_a or adapter_name not in lora_b:
            continue
        a_abs = lora_a[adapter_name].weight.detach().float().abs().sum().item()
        b_abs = lora_b[adapter_name].weight.detach().float().abs().sum().item()
        stats.total_a_abs += a_abs
        stats.total_b_abs += b_abs
        stats.nonzero_a += int(a_abs > 0)
        stats.nonzero_b += int(b_abs > 0)

    print(f"nonzero_A_layers: {stats.nonzero_a}/{stats.count}")
    print(f"nonzero_B_layers: {stats.nonzero_b}/{stats.count}")
    print(f"total_A_abs: {stats.total_a_abs:.6e}")
    print(f"total_B_abs: {stats.total_b_abs:.6e}")
    return stats


def load_lora_current_v03(transformer: torch.nn.Module, args: argparse.Namespace, adapter_name: str) -> None:
    source, source_kwargs = resolve_lora_for_direct_loader(args.lora_path, args.weight_name)
    kwargs = {"adapter_name": adapter_name, "prefix": "transformer", **source_kwargs}
    print(f"direct_load_call: source={source} kwargs={kwargs}")
    transformer.load_lora_adapter(source, **kwargs)

    if adapter_name not in getattr(transformer, "peft_config", {}):
        kwargs["prefix"] = None
        print(f"direct_retry_call: source={source} kwargs={kwargs}")
        transformer.load_lora_adapter(source, **kwargs)

    if adapter_name in getattr(transformer, "peft_config", {}):
        transformer.set_adapters(adapter_name, weights=args.lora_scale)


def load_lora_official_wan(transformer: torch.nn.Module, args: argparse.Namespace, adapter_name: str) -> None:
    from diffusers.loaders.lora_pipeline import WanLoraLoaderMixin

    source, source_kwargs = resolve_lora_for_official_loader(args.lora_path, args.weight_name)
    kwargs: dict[str, Any] = {
        "local_files_only": args.local_files_only,
        "return_lora_metadata": True,
        **source_kwargs,
    }

    print(f"official_lora_state_dict_call: source={source} kwargs={kwargs}")
    result = WanLoraLoaderMixin.lora_state_dict(source, **kwargs)
    if isinstance(result, tuple) and len(result) == 2:
        state_dict, metadata = result
    elif isinstance(result, tuple) and len(result) == 3:
        state_dict, _, metadata = result
    else:
        state_dict, metadata = result, None

    if hasattr(WanLoraLoaderMixin, "_maybe_expand_t2v_lora_for_i2v"):
        state_dict = WanLoraLoaderMixin._maybe_expand_t2v_lora_for_i2v(transformer=transformer, state_dict=state_dict)

    print(f"official_converted_state_dict_tensors: {len(state_dict)}")
    for key in list(state_dict)[: args.max_keys]:
        print(f"  official_key={key} shape={tuple(state_dict[key].shape)}")

    WanLoraLoaderMixin.load_lora_into_transformer(
        state_dict,
        transformer=transformer,
        adapter_name=adapter_name,
        metadata=metadata,
        low_cpu_mem_usage=False,
    )
    if adapter_name in getattr(transformer, "peft_config", {}):
        transformer.set_adapters(adapter_name, weights=args.lora_scale)


def tensor_from_model_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    if hasattr(output, "sample"):
        return output.sample
    raise TypeError(f"Unsupported transformer output type: {type(output)}")


def run_forward_difference(transformer: torch.nn.Module, args: argparse.Namespace, adapter_name: str) -> None:
    log_section("Forward difference check")
    device = next(transformer.parameters()).device
    latent_dtype = transformer.patch_embedding.weight.dtype
    text_dtype = transformer.condition_embedder.text_embedder.linear_1.weight.dtype
    config = transformer.config
    patch_size = tuple(config.patch_size)
    frames = max(args.forward_frames, patch_size[0])
    height = max(args.forward_height, patch_size[1])
    width = max(args.forward_width, patch_size[2])
    if frames % patch_size[0] != 0 or height % patch_size[1] != 0 or width % patch_size[2] != 0:
        raise ValueError(f"Forward dimensions must be divisible by patch_size={patch_size}")

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    hidden_states = torch.randn(
        1,
        config.in_channels,
        frames,
        height,
        width,
        generator=generator,
        device=device,
        dtype=latent_dtype,
    )
    encoder_hidden_states = torch.randn(
        1,
        512,
        config.text_dim,
        generator=generator,
        device=device,
        dtype=text_dtype,
    )
    timestep = torch.full((1,), 0.5, device=device, dtype=torch.float32)

    with torch.no_grad():
        transformer.set_adapters(adapter_name, weights=0.0)
        out0 = tensor_from_model_output(
            transformer(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )
        )

        transformer.set_adapters(adapter_name, weights=args.lora_scale)
        out1 = tensor_from_model_output(
            transformer(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )
        )

    diff = (out1.float() - out0.float()).abs()
    print(f"forward_shape: {tuple(out0.shape)}")
    print(f"scale0_vs_scale{args.lora_scale}_max_abs_diff: {diff.max().item():.8e}")
    print(f"scale0_vs_scale{args.lora_scale}_mean_abs_diff: {diff.mean().item():.8e}")
    if diff.max().item() == 0.0:
        fail("Forward output is identical between scale=0 and requested scale. Adapter is not affecting this forward.")
    else:
        ok("Forward output changes when LoRA scale changes.")


def run_loader_mode(args: argparse.Namespace, mode: str) -> LayerStats | None:
    adapter_name = f"{LORA_ADAPTER_NAME}_{mode}"
    log_section(f"Loader mode: {mode}")
    transformer = None
    try:
        transformer = load_transformer(args.model_path, parse_dtype(args.dtype), args.device, args.local_files_only)
        if mode == "current_v03":
            load_lora_current_v03(transformer, args, adapter_name)
        elif mode == "official_wan":
            load_lora_official_wan(transformer, args, adapter_name)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        stats = inspect_loaded_layers(transformer, adapter_name, args.max_layers)
        if stats.count == 0:
            fail(f"{mode}: no PEFT LoRA layers were injected.")
        elif stats.nonzero_b == 0:
            fail(f"{mode}: LoRA layers exist, but all inspected lora_B weights are zero.")
        else:
            ok(f"{mode}: injected {stats.count} LoRA layers with non-zero B weights in {stats.nonzero_b} layers.")

        if args.forward_check:
            try:
                run_forward_difference(transformer, args, adapter_name)
            except Exception as exc:
                fail(f"{mode} forward_check: {type(exc).__name__}: {exc}")
                if args.verbose:
                    traceback.print_exc()
        return stats
    except Exception as exc:
        fail(f"{mode}: {type(exc).__name__}: {exc}")
        if args.verbose:
            traceback.print_exc()
        return None
    finally:
        if transformer is not None:
            del transformer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Wan Diffusers LoRA loading and effect.")
    parser.add_argument("--model_path", required=True, help="Local Wan Diffusers base model path.")
    parser.add_argument("--lora_path", required=True, help="Local LoRA file/directory, or cached repo id.")
    parser.add_argument("--weight_name", default=None, help="LoRA weight filename when --lora_path is a directory/repo.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "fp32", "bfloat16", "bf16", "float16", "fp16"])
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--loader", choices=["current_v03", "official_wan", "both"], default="both")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_model_load", action="store_true", help="Only inspect raw LoRA keys; do not load base model.")
    parser.add_argument("--forward_check", action="store_true", help="Run a tiny random transformer forward at scale 0 and scale.")
    parser.add_argument("--forward_frames", type=int, default=1)
    parser.add_argument("--forward_height", type=int, default=4)
    parser.add_argument("--forward_width", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_keys", type=int, default=30)
    parser.add_argument("--max_layers", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_section("Environment")
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    try:
        import diffusers

        print(f"diffusers: {diffusers.__version__}")
    except Exception as exc:
        fail(f"diffusers import failed: {exc}")
        return 2
    try:
        import peft

        print(f"peft: {peft.__version__}")
    except Exception as exc:
        fail(f"peft import failed: {exc}")
        return 2
    print(f"model_path: {args.model_path}")
    print(f"lora_path: {args.lora_path}")
    print(f"weight_name: {args.weight_name}")
    print(f"device: {args.device}")
    print(f"dtype: {args.dtype}")
    print(f"local_files_only: {args.local_files_only}")

    summarize_raw_lora(args.lora_path, args.weight_name, args.max_keys)

    if args.skip_model_load:
        return 0

    modes = ["current_v03", "official_wan"] if args.loader == "both" else [args.loader]
    results = {mode: run_loader_mode(args, mode) for mode in modes}

    if args.loader == "both":
        log_section("Comparison")
        cur = results.get("current_v03")
        off = results.get("official_wan")
        cur_count = cur.count if cur is not None else 0
        off_count = off.count if off is not None else 0
        print(f"current_v03_layers: {cur_count}")
        print(f"official_wan_layers: {off_count}")
        if cur_count == 0 and off_count > 0:
            fail("Current v0.3 direct loader failed while official Wan loader injected layers.")
            print("Likely root cause: LoRA checkpoint needs Diffusers Wan-specific key conversion.")
        elif cur_count > 0 and off_count > 0:
            ok("Both loader paths injected LoRA layers. Use forward_check or generated video comparison next.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
