"""Protect primary CNN run records from accidental overwrite or false reuse."""

from dataclasses import asdict
import json
from pathlib import Path


def metadata(cfg, *, eval_every: int, device: str, data_root: Path) -> dict:
    return {**asdict(cfg), "eval_every": eval_every, "device": device,
            "data_root": str(data_root.resolve())}


def check_existing(path: Path, expected: dict, *, skip_existing: bool) -> bool:
    if not path.exists():
        return False
    if not skip_existing:
        raise FileExistsError(f"refusing to overwrite an existing run: {path}")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(saved, dict) or any(
        saved.get(name) != value for name, value in expected.items()
    ):
        raise ValueError(f"existing run does not match every requested parameter: {path}")
    return True


def save_exclusive(path: Path, result: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
