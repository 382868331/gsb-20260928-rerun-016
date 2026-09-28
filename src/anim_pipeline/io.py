from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from .models import Animation, Mapping, Skeleton


def read_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_skeleton(path: str | Path) -> Skeleton:
    return Skeleton.from_dict(read_json(path))


def load_animation(path: str | Path) -> Animation:
    return Animation.from_dict(read_json(path))


def load_mapping(path: str | Path) -> Mapping:
    return Mapping.from_dict(read_json(path))


def atomic_write_json(path: str | Path, data: dict, *, overwrite: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing resource: {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
