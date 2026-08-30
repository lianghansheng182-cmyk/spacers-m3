"""Software-level SEU-equivalent fault injection for LoRA adapter tensors.

This module only mutates caller-provided adapter copies. It does not touch the
frozen LoRA v1 asset unless the caller explicitly passes that path as output.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


FLOAT_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    view = tensor.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(view.numpy().tobytes()).hexdigest()


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class TensorGroup:
    tensor_name: str
    layer: str
    projection: str
    lora_side: str
    domain: str


def classify_tensor(tensor_name: str) -> TensorGroup:
    parts = tensor_name.split(".")
    layer = "unknown"
    for index, part in enumerate(parts):
        if part == "layers" and index + 1 < len(parts):
            layer = parts[index + 1]
            break
    projection = "unknown"
    for candidate in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        if candidate in tensor_name:
            projection = candidate
            break
    if "lora_A" in tensor_name:
        lora_side = "A"
    elif "lora_B" in tensor_name:
        lora_side = "B"
    else:
        lora_side = "unknown"
    domain = "vision" if any(token in tensor_name for token in ("visual", "vision", "merger")) else "language"
    return TensorGroup(tensor_name, layer, projection, lora_side, domain)


def adapter_tensor_inventory(adapter_model_path: Path) -> list[dict[str, Any]]:
    tensors = load_file(str(adapter_model_path), device="cpu")
    rows: list[dict[str, Any]] = []
    for name, tensor in sorted(tensors.items()):
        group = classify_tensor(name)
        rows.append(
            {
                "tensor_name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "numel": int(tensor.numel()),
                "bytes": int(tensor.numel() * tensor.element_size()),
                "layer": group.layer,
                "projection": group.projection,
                "lora_side": group.lora_side,
                "domain": group.domain,
                "sha256": sha256_tensor(tensor),
            }
        )
    return rows


def _candidate_tensor_names(
    tensors: dict[str, torch.Tensor],
    group_filter: dict[str, str] | None,
) -> list[str]:
    names = []
    for name, tensor in tensors.items():
        if tensor.numel() == 0 or tensor.dtype not in FLOAT_DTYPES:
            continue
        group = classify_tensor(name)
        if group_filter:
            if any(getattr(group, key) != value for key, value in group_filter.items()):
                continue
        names.append(name)
    if not names:
        raise ValueError(f"No tensors matched group_filter={group_filter}")
    return sorted(names)


def inject_adapter_bitflips(
    source_adapter_dir: Path,
    output_adapter_dir: Path,
    *,
    seed: int,
    fault_count: int,
    level: str,
    group_filter: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Copy an adapter directory and flip bits in the copied safetensors file."""

    source_adapter_dir = source_adapter_dir.resolve()
    output_adapter_dir = output_adapter_dir.resolve()
    if output_adapter_dir.exists():
        shutil.rmtree(output_adapter_dir)
    shutil.copytree(source_adapter_dir, output_adapter_dir)

    adapter_path = output_adapter_dir / "adapter_model.safetensors"
    original_file_sha256 = sha256_file(adapter_path)
    tensors = load_file(str(adapter_path), device="cpu")
    candidates = _candidate_tensor_names(tensors, group_filter)
    rng = random.Random(seed)
    faults: list[dict[str, Any]] = []
    used: set[tuple[str, int, int]] = set()

    for fault_index in range(fault_count):
        tensor_name = rng.choice(candidates)
        tensor = tensors[tensor_name].contiguous()
        flat = tensor.view(-1)
        element_index = rng.randrange(flat.numel())
        bit_position = rng.randrange(tensor.element_size() * 8)
        dedupe_key = (tensor_name, element_index, bit_position)
        retry = 0
        while dedupe_key in used and retry < 100:
            tensor_name = rng.choice(candidates)
            tensor = tensors[tensor_name].contiguous()
            flat = tensor.view(-1)
            element_index = rng.randrange(flat.numel())
            bit_position = rng.randrange(tensor.element_size() * 8)
            dedupe_key = (tensor_name, element_index, bit_position)
            retry += 1
        used.add(dedupe_key)

        before_value = str(flat[element_index].item())
        before_tensor_sha256 = sha256_tensor(tensor)
        byte_offset = element_index * tensor.element_size() + bit_position // 8
        bit_in_byte = bit_position % 8
        byte_view = tensor.view(torch.uint8).view(-1)
        original_byte = int(byte_view[byte_offset].item())
        byte_view[byte_offset] = original_byte ^ (1 << bit_in_byte)
        after_value = str(flat[element_index].item())
        after_tensor_sha256 = sha256_tensor(tensor)
        tensors[tensor_name] = tensor

        group = classify_tensor(tensor_name)
        faults.append(
            {
                "fault_index": fault_index,
                "fault_type": "software_level_seu_equivalent_lora_tensor_bitflip",
                "level": level,
                "seed": seed,
                "affected_tensor": tensor_name,
                "tensor_group": {
                    "layer": group.layer,
                    "projection": group.projection,
                    "lora_side": group.lora_side,
                    "domain": group.domain,
                },
                "element_index": element_index,
                "bit_position": bit_position,
                "byte_offset_within_tensor": byte_offset,
                "original_byte": original_byte,
                "corrupted_byte": int(byte_view[byte_offset].item()),
                "original_value": before_value,
                "corrupted_value": after_value,
                "original_tensor_sha256": before_tensor_sha256,
                "corrupted_tensor_sha256": after_tensor_sha256,
            }
        )

    save_file(tensors, str(adapter_path))
    corrupted_file_sha256 = sha256_file(adapter_path)
    manifest = {
        "created_at_unix": time.time(),
        "source_adapter_dir": str(source_adapter_dir),
        "output_adapter_dir": str(output_adapter_dir),
        "adapter_model": str(adapter_path),
        "fault_model": "software-level SEU-equivalent fault injection",
        "note": "This is not a real space radiation or space-qualified hardware experiment.",
        "seed": seed,
        "level": level,
        "fault_count": fault_count,
        "group_filter": group_filter or {},
        "original_file_sha256": original_file_sha256,
        "corrupted_file_sha256": corrupted_file_sha256,
        "faults": faults,
    }
    write_json(output_adapter_dir / "fault_manifest.json", manifest)
    return manifest
