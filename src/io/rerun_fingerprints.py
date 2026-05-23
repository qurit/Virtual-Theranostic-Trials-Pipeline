"""
JSON normalization and filesystem fingerprint helpers for rerun safety.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


def normalize_jsonable(value: Any) -> Any:
    """Convert values into a stable JSON-serializable structure."""
    if isinstance(value, dict):
        return {
            str(k): normalize_jsonable(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [normalize_jsonable(v) for v in value]
    if isinstance(value, set):
        return [normalize_jsonable(v) for v in sorted(value, key=lambda x: repr(x))]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            pass
    return value


def json_digest(value: Any) -> str:
    """Return a SHA256 digest of a normalized JSON-compatible value."""
    payload = json.dumps(
        normalize_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_file_into(hasher: Any, file_path: Path) -> int:
    """Update ``hasher`` with file bytes and return the byte count."""
    total = 0
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            total += len(chunk)
            hasher.update(chunk)
    return total


def fingerprint_file(path: str | Path) -> Dict[str, Any]:
    """Fingerprint a single file by content."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found for fingerprinting: {file_path}")

    hasher = hashlib.sha256()
    size_bytes = _hash_file_into(hasher, file_path)
    return {
        "kind": "file",
        "path": str(file_path.resolve()),
        "sha256": hasher.hexdigest(),
        "size_bytes": int(size_bytes),
    }


def fingerprint_path(path: str | Path) -> Dict[str, Any]:
    """
    Fingerprint a file or directory by content.

    Directories are hashed recursively using sorted relative file paths plus
    each file's byte content.
    """
    p = Path(path)
    if p.is_file():
        return fingerprint_file(p)
    if not p.is_dir():
        raise FileNotFoundError(f"Path not found for fingerprinting: {p}")

    files = sorted(fp for fp in p.rglob("*") if fp.is_file())
    hasher = hashlib.sha256()
    total_bytes = 0
    for fp in files:
        rel = fp.relative_to(p).as_posix().encode("utf-8")
        size = fp.stat().st_size
        hasher.update(b"FILE\0")
        hasher.update(rel)
        hasher.update(b"\0")
        hasher.update(str(size).encode("utf-8"))
        hasher.update(b"\0")
        total_bytes += _hash_file_into(hasher, fp)

    return {
        "kind": "directory",
        "path": str(p.resolve()),
        "sha256": hasher.hexdigest(),
        "file_count": len(files),
        "size_bytes": int(total_bytes),
    }


def fingerprint_optional_file(path: str | Path | None) -> Optional[Dict[str, Any]]:
    """Return a file fingerprint when the path exists, else ``None``."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return fingerprint_file(p)


def fingerprints_equal(lhs: Any, rhs: Any) -> bool:
    """Return True when two fingerprint dicts describe the same content."""
    lhs_n = normalize_jsonable(lhs)
    rhs_n = normalize_jsonable(rhs)
    if not isinstance(lhs_n, dict) or not isinstance(rhs_n, dict):
        return False
    return (
        lhs_n.get("kind") == rhs_n.get("kind")
        and lhs_n.get("sha256") == rhs_n.get("sha256")
    )


def flatten_paths(value: Any) -> List[str]:
    """Flatten nested dict/list path containers into one list of path strings."""
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [str(value)]
    if isinstance(value, Mapping):
        out: List[str] = []
        for item in value.values():
            out.extend(flatten_paths(item))
        return out
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        out = []
        for item in value:
            out.extend(flatten_paths(item))
        return out
    return []


def any_existing_paths(paths: Any) -> bool:
    """Return True when any path in ``paths`` currently exists."""
    return any(Path(p).exists() for p in flatten_paths(paths))
