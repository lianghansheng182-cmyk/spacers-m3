"""Integrity manifests and startup checks for OrbitGuard-RS."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .fault_injection import sha256_file


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


@dataclass(frozen=True)
class IntegrityResult:
    status: str
    check_latency_s: float
    checked_files: int
    mismatches: list[dict[str, Any]]
    missing: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "check_latency_s": round(self.check_latency_s, 6),
            "checked_files": self.checked_files,
            "mismatches": self.mismatches,
            "missing": self.missing,
        }


def collect_asset(path: Path, role: str, repo_root: Path) -> list[dict[str, Any]]:
    path = path.resolve()
    rows: list[dict[str, Any]] = []
    if not path.exists():
        rows.append({"role": role, "path": str(path), "exists": False})
        return rows
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for file_path in files:
        stat = file_path.stat()
        try:
            rel = str(file_path.relative_to(repo_root))
        except ValueError:
            rel = str(file_path)
        rows.append(
            {
                "role": role,
                "path": str(file_path),
                "relative_path": rel,
                "exists": True,
                "size_bytes": stat.st_size,
                "sha256": sha256_file(file_path),
            }
        )
    return rows


def build_integrity_manifest(
    *,
    repo_root: Path,
    assets: list[tuple[str, Path]],
    output_path: Path,
    version: str,
    backup_dir: Path | None = None,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for role, asset_path in assets:
        entries.extend(collect_asset(asset_path, role, repo_root))
    manifest = {
        "name": "OrbitGuard-RS integrity manifest",
        "version": version,
        "created_at_unix": time.time(),
        "git_commit": git_commit(repo_root),
        "repo_root": str(repo_root.resolve()),
        "backup_dir": str(backup_dir.resolve()) if backup_dir else None,
        "assets": entries,
    }
    write_json(output_path, manifest)
    return manifest


def verify_integrity(manifest_path: Path, root_override: Path | None = None) -> IntegrityResult:
    start = time.perf_counter()
    manifest = load_json(manifest_path)
    repo_root = root_override.resolve() if root_override else Path(manifest["repo_root"]).resolve()
    mismatches: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    checked = 0
    for entry in manifest["assets"]:
        if not entry.get("exists", True):
            missing.append(entry)
            continue
        path = Path(entry["path"])
        if not path.is_absolute():
            path = repo_root / path
        if not path.exists():
            missing.append(entry)
            continue
        checked += 1
        current_size = path.stat().st_size
        current_sha256 = sha256_file(path)
        if current_size != entry.get("size_bytes") or current_sha256 != entry.get("sha256"):
            mismatches.append(
                {
                    "role": entry.get("role"),
                    "path": str(path),
                    "expected_size_bytes": entry.get("size_bytes"),
                    "actual_size_bytes": current_size,
                    "expected_sha256": entry.get("sha256"),
                    "actual_sha256": current_sha256,
                }
            )
    latency = time.perf_counter() - start
    status = "PASS" if not mismatches and not missing else "DETECTED"
    return IntegrityResult(status, latency, checked, mismatches, missing)


def materialize_backup(manifest: dict[str, Any], backup_dir: Path) -> dict[str, Any]:
    backup_dir = backup_dir.resolve()
    copied: list[dict[str, Any]] = []
    skipped_same_file: list[dict[str, Any]] = []
    for entry in manifest["assets"]:
        source = Path(entry["path"])
        if not source.exists() or not source.is_file():
            continue
        relative_path = Path(entry["relative_path"])
        if relative_path.is_absolute():
            relative_path = Path("external_assets") / str(relative_path).lstrip("/")
        destination = backup_dir / relative_path
        if source.resolve() == destination.resolve():
            skipped_same_file.append(
                {
                    "role": entry.get("role"),
                    "source": str(source),
                    "backup": str(destination),
                    "reason": "source_and_backup_resolve_to_same_file",
                }
            )
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(
            {
                "role": entry.get("role"),
                "source": str(source),
                "backup": str(destination),
                "size_bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
    payload = {
        "created_at_unix": time.time(),
        "backup_dir": str(backup_dir),
        "copied_files": copied,
        "skipped_same_file": skipped_same_file,
        "total_backup_bytes": sum(row["size_bytes"] for row in copied),
    }
    write_json(backup_dir / "backup_manifest.json", payload)
    return payload
