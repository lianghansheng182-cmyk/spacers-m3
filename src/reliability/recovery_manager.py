"""Automatic recovery helpers for OrbitGuard-RS."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from .integrity_guard import verify_integrity


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class RecoveryManager:
    """Restore corrupted deployed files from a trusted read-only backup copy."""

    def __init__(self, manifest_path: Path, backup_dir: Path):
        self.manifest_path = manifest_path.resolve()
        self.backup_dir = backup_dir.resolve()

    def recover(self) -> dict[str, Any]:
        start = time.perf_counter()
        before = verify_integrity(self.manifest_path).as_dict()
        restored: list[dict[str, Any]] = []
        if before["status"] == "DETECTED":
            for mismatch in before["mismatches"]:
                target = Path(mismatch["path"]).resolve()
                try:
                    rel = target.relative_to(Path(json.loads(self.manifest_path.read_text())["repo_root"]).resolve())
                except ValueError:
                    rel = Path(target.name)
                backup = self.backup_dir / rel
                if not backup.exists():
                    restored.append({"target": str(target), "backup": str(backup), "restored": False, "reason": "backup missing"})
                    continue
                shutil.copy2(backup, target)
                restored.append({"target": str(target), "backup": str(backup), "restored": True})
        after = verify_integrity(self.manifest_path).as_dict()
        return {
            "detection_status": before["status"],
            "pre_recovery": before,
            "restored_files": restored,
            "post_recovery": after,
            "recovery_success": before["status"] == "DETECTED" and after["status"] == "PASS",
            "recovery_latency_s": round(time.perf_counter() - start, 6),
        }
