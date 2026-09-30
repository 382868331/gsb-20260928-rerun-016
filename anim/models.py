from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from .math3d import Quat, Vec, quat_from_axes


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str | None
    offset: Vec
    bind_rotation: Quat | None
    rotation_limits: dict[str, tuple[float, float]] | None


class Skeleton:
    def __init__(self, data: dict[str, Any]):
        self.name = data["name"]
        self.root = data["root"]
        self.joints: dict[str, Joint] = {}
        for name, item in data["joints"].items():
            parent = item.get("parent")
            axes = item.get("bind_axes")
            bind_rotation = None
            if axes is not None:
                bind_rotation = quat_from_axes(axes["x"], axes["y"], axes["z"])
            limits = None
            if item.get("rotation_limits"):
                limits = {axis: tuple(values) for axis, values in item["rotation_limits"].items()}
            self.joints[name] = Joint(name, parent, tuple(item["offset"]), bind_rotation, limits)
        if self.root not in self.joints:
            raise ValueError(f"skeleton {self.name} references unknown root {self.root}")
        self._validate_hierarchy()

    def _validate_hierarchy(self) -> None:
        for name, joint in self.joints.items():
            if name == self.root:
                if joint.parent is not None:
                    raise ValueError(f"root joint {name} must have null parent")
                continue
            if joint.parent is None:
                raise ValueError(f"non-root joint {name} requires a parent")
            if joint.parent not in self.joints:
                raise ValueError(f"joint {name} references unknown parent {joint.parent}")

    def children(self, name: str) -> list[str]:
        return [joint.name for joint in self.joints.values() if joint.parent == name]

    def ordered(self) -> list[str]:
        result: list[str] = []

        def visit(name: str) -> None:
            result.append(name)
            for child in self.children(name):
                visit(child)

        visit(self.root)
        return result


class Motion:
    def __init__(self, data: dict[str, Any]):
        self.name = data["name"]
        self.skeleton = data["skeleton"]
        self.frame_rate = float(data["frame_rate"])
        self.frames = data["frames"]
        self.contacts = data.get("contacts", [])
        self.local_revisions = data.get("local_revisions", [])
        if not self.frames:
            raise ValueError("motion must contain at least one frame")
        for index, frame in enumerate(self.frames):
            if "root_translation" not in frame or "rotations" not in frame:
                raise ValueError(f"frame {index} requires root_translation and rotations")


class Mapping:
    def __init__(self, data: dict[str, Any]):
        self.name = data["name"]
        self.version = data.get("version", "unversioned")
        self.source_skeleton = data["source_skeleton"]
        self.target_skeleton = data["target_skeleton"]
        self.entries = data["joint_mappings"]

    def pairs(self) -> dict[str, str | None]:
        return {item["source"]: item.get("target") for item in self.entries}

    def required_pairs(self) -> dict[str, str]:
        result = {}
        for item in self.entries:
            if item.get("required", True) and item.get("target"):
                result[item["source"]] = item["target"]
        return result


class Batch:
    def __init__(self, data: dict[str, Any]):
        self.name = data["name"]
        self.jobs = data["jobs"]


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_skeleton(path: str | Path) -> Skeleton:
    return Skeleton(load_json(path))


def load_motion(path: str | Path) -> Motion:
    return Motion(load_json(path))


def load_mapping(path: str | Path) -> Mapping:
    return Mapping(load_json(path))


def load_batch(path: str | Path) -> Batch:
    return Batch(load_json(path))


def atomic_write_json(path: str | Path, value: Any, overwrite: bool = False) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing resource: {destination}")
    fd = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent, delete=False) as handle:
            fd = handle.name
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(fd, destination)
        fd = None
    finally:
        if fd is not None:
            try:
                os.unlink(fd)
            except FileNotFoundError:
                pass
