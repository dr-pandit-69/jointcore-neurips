"""Small deterministic utilities shared by the JointCore exact pilot."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    return int(sha256_text("\x1f".join(str(part) for part in parts))[:16], 16)


def stable_rng(*parts: object) -> random.Random:
    return random.Random(stable_seed(*parts))


def unit_interval(*parts: object) -> float:
    return stable_seed(*parts) / float(0xFFFFFFFFFFFFFFFF)


def config_hash(config: dict[str, Any]) -> str:
    return sha256_text(canonical_json(config))


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return loaded


def portable_path(path: Path, project_root: Path) -> str:
    """Return a project-relative path, or an absolute path for external output."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def git_metadata(repo_root: Path) -> dict[str, object]:
    def invoke(*args: str) -> tuple[int, str]:
        result = subprocess.run(
            ["git", *args], cwd=repo_root, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, check=False,
        )
        return result.returncode, result.stdout.strip()

    code, commit = invoke("rev-parse", "HEAD")
    _, dirty = invoke("status", "--porcelain")
    return {
        "git_commit": commit if code == 0 else "UNCOMMITTED_INITIAL_REPOSITORY",
        "git_dirty": bool(dirty),
    }
