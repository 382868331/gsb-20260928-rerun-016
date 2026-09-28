from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .math3d import qlist, quat, vec3


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str | None
    translation: np.ndarray
    rotation: np.ndarray
    limits_rad: tuple[tuple[float, float], ...] | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "Joint":
        limits = data.get("joint_limits", {}).get("rotation_xyz_rad")
        return cls(
            str(data["name"]),
            data.get("parent"),
            vec3(data.get("translation", [0, 0, 0])),
            qlist(data.get("rotation", [0, 0, 0, 1])),
            tuple((float(a), float(b)) for a, b in limits) if limits else None,
        )


@dataclass
class Skeleton:
    name: str
    joints: dict[str, Joint]
    root_name: str

    @classmethod
    def from_dict(cls, data: dict) -> "Skeleton":
        joints = {item["name"]: Joint.from_dict(item) for item in data["joints"]}
        root = data.get("root")
        if root is None:
            roots = [j.name for j in joints.values() if j.parent is None]
            if len(roots) != 1:
                raise ValueError("skeleton must have exactly one root")
            root = roots[0]
        skeleton = cls(str(data.get("name", "skeleton")), joints, root)
        skeleton.validate_hierarchy()
        return skeleton

    def validate_hierarchy(self) -> None:
        if self.root_name not in self.joints:
            raise ValueError(f"unknown root joint {self.root_name!r}")
        for name, joint in self.joints.items():
            if joint.parent is not None and joint.parent not in self.joints:
                raise ValueError(f"joint {name!r} references missing parent {joint.parent!r}")

    def children_order(self) -> list[str]:
        ordered: list[str] = []

        def visit(name: str) -> None:
            ordered.append(name)
            for child_name, joint in self.joints.items():
                if joint.parent == name:
                    visit(child_name)

        visit(self.root_name)
        if len(ordered) != len(self.joints):
            raise ValueError("skeleton hierarchy contains a cycle or detached joint")
        return ordered


@dataclass
class Animation:
    name: str
    skeleton_name: str
    frame_time: float
    root_translations: list[np.ndarray]
    local_rotations: list[dict[str, np.ndarray]]
    metadata: dict

    @classmethod
    def from_dict(cls, data: dict) -> "Animation":
        frames = data["frames"]
        return cls(
            str(data.get("name", "animation")),
            str(data["skeleton"]),
            float(data.get("frame_time", 1.0 / 30.0)),
            [vec3(frame["root_translation"]) for frame in frames],
            [{k: qlist(v) for k, v in frame["local_rotations"].items()} for frame in frames],
            dict(data.get("metadata", {})),
        )

    @property
    def frame_count(self) -> int:
        return len(self.root_translations)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "skeleton": self.skeleton_name,
            "frame_time": self.frame_time,
            "metadata": self.metadata,
            "frames": [
                {
                    "index": index,
                    "time": index * self.frame_time,
                    "root_translation": [float(x) for x in translation],
                    "local_rotations": {
                        name: [float(x) for x in rotation]
                        for name, rotation in sorted(rotations.items())
                    },
                }
                for index, (translation, rotations) in enumerate(
                    zip(self.root_translations, self.local_rotations, strict=True)
                )
            ],
        }


@dataclass(frozen=True)
class Mapping:
    name: str
    source_skeleton: str
    target_skeleton: str
    pairs: dict[str, str | None]
    required_targets: frozenset[str]
    scale_mode: str

    @classmethod
    def from_dict(cls, data: dict) -> "Mapping":
        pairs = {}
        required = []
        for item in data["joint_pairs"]:
            source = item.get("source")
            target = item.get("target")
            if source is None and target is None:
                raise ValueError("a mapping pair must reference a source or target")
            if source is not None:
                pairs[str(source)] = None if target is None else str(target)
            if target is not None and item.get("required", True):
                required.append(str(target))
        return cls(
            str(data.get("name", "mapping")),
            str(data["source_skeleton"]),
            str(data["target_skeleton"]),
            pairs,
            frozenset(required),
            str(data.get("scale_mode", "leg_length")),
        )

    @property
    def inverse_pairs(self) -> dict[str, str]:
        return {target: source for source, target in self.pairs.items() if target is not None}


def rest_rotation_dict(skeleton: Skeleton) -> dict[str, np.ndarray]:
    return {name: quat() if False else skeleton.joints[name].rotation.copy() for name in skeleton.joints}
