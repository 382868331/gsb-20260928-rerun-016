from __future__ import annotations

import numpy as np

from .math3d import axis_angle, inverse, multiply, rotation_between


def simple_skeleton(name: str, leg_length: float, width: float, root_height: float) -> dict:
    upper = leg_length * 0.5
    lower = leg_length * 0.5
    foot = 0.12
    side = width / 2
    spine_rotation = [0, 0, 0, 1] if name == "biped_source_a" else [0, 0, 0.0871557427, 0.9961946981]
    result = {
        "name": name,
        "root": "pelvis",
        "joints": [
            {"name": "pelvis", "parent": None, "translation": [0, root_height, 0]},
            {"name": "spine", "parent": "pelvis", "translation": [0, 0.22, 0], "rotation": spine_rotation},
            {
                "name": "left_upleg",
                "parent": "pelvis",
                "translation": [side, 0, 0],
                "joint_limits": {"rotation_xyz_rad": [[-2.4, 2.4], [-2.4, 2.4], [-2.4, 2.4]]},
            },
            {
                "name": "right_upleg",
                "parent": "pelvis",
                "translation": [-side, 0, 0],
                "joint_limits": {"rotation_xyz_rad": [[-2.4, 2.4], [-2.4, 2.4], [-2.4, 2.4]]},
            },
            {"name": "left_leg", "parent": "left_upleg", "translation": [0, -upper, 0]},
            {"name": "right_leg", "parent": "right_upleg", "translation": [0, -upper, 0]},
            {"name": "left_foot", "parent": "left_leg", "translation": [0, -lower, 0]},
            {"name": "right_foot", "parent": "right_leg", "translation": [0, -lower, 0]},
            {"name": "left_toe", "parent": "left_foot", "translation": [foot, 0, 0]},
            {"name": "right_toe", "parent": "right_foot", "translation": [foot, 0, 0]},
        ],
    }
    if name == "biped_source_a":
        result["joints"].append(
            {"name": "source_ponytail", "parent": "spine", "translation": [0, 0.12, -0.03]}
        )
    if name == "biped_target_b":
        result["joints"].append(
            {"name": "target_tail", "parent": "pelvis", "translation": [0, -0.05, -0.12]}
        )
    return result


def _solve_leg_ik(hip_position, foot_position, upper, lower, forward=np.array([1.0, 0, 0])):
    to_foot = foot_position - hip_position
    distance = min(float(np.linalg.norm(to_foot)), upper + lower - 1e-6)
    direction = to_foot / np.linalg.norm(to_foot)
    normal = np.cross(direction, forward)
    if np.linalg.norm(normal) < 1e-6:
        normal = np.array([0.0, 0.0, 1.0])
    normal /= np.linalg.norm(normal)
    angle = np.arccos(np.clip((upper * upper + distance * distance - lower * lower) / (2 * upper * distance), -1, 1))
    knee = hip_position + upper * (np.cos(angle) * direction + np.sin(angle) * normal)
    hip_rotation = rotation_between(np.array([0.0, -1.0, 0.0]), knee - hip_position)
    knee_rotation = rotation_between(np.array([0.0, -1.0, 0.0]), foot_position - knee)
    return hip_rotation, knee_rotation, knee


def walk_animation(skeleton_name: str, frame_count: int = 18) -> dict:
    upper = 0.45
    lower = 0.45
    side = 0.08
    frame_time = 1 / 30
    frames = []
    stride = 0.42
    for i in range(frame_count):
        phase = 2 * np.pi * i / max(frame_count - 1, 1)
        root_x = stride * i / max(frame_count - 1, 1)
        root = np.array([root_x, 0.02 * np.sin(phase), 0.0])
        rotations = {
            "pelvis": [0, 0, 0, 1],
            "spine": [0, 0, 0, 1],
            "left_toe": [0, 0, 0, 1],
            "right_toe": [0, 0, 0, 1],
        }
        for side_name, hip_x, phase_offset in (
            ("left", side, 0.0),
            ("right", -side, np.pi),
        ):
            foot_phase = (phase + phase_offset) % (2 * np.pi)
            if phase_offset == 0.0:
                if foot_phase <= np.pi:
                    swing_phase = foot_phase / np.pi
                    foot_x = stride / 2 * (swing_phase * swing_phase * (3 - 2 * swing_phase))
                    lift = 0.10 * np.sin(np.pi * swing_phase)
                else:
                    foot_x = stride / 2
                    lift = 0.0
            else:
                if foot_phase <= np.pi:
                    foot_x = 0.0
                    lift = 0.0
                else:
                    swing_phase = (foot_phase - np.pi) / np.pi
                    foot_x = stride * (swing_phase * swing_phase * (3 - 2 * swing_phase))
                    lift = 0.10 * np.sin(np.pi * swing_phase)
            foot = np.array([foot_x, lift, 0.0])
            hip_rotation, knee_rotation, _ = _solve_leg_ik(
                root + np.array([hip_x, 0, 0]), foot, upper, lower
            )
            rotations[f"{side_name}_upleg"] = list(hip_rotation)
            rotations[f"{side_name}_leg"] = list(multiply(inverse(hip_rotation), knee_rotation))
            rotations[f"{side_name}_foot"] = [0, 0, 0, 1]
        frames.append(
            {
                "index": i,
                "time": i * frame_time,
                "root_translation": [float(x) for x in root],
                "local_rotations": {k: [float(x) for x in v] for k, v in rotations.items()},
            }
        )
    return {
        "name": "sample_walk",
        "skeleton": skeleton_name,
        "frame_time": frame_time,
        "metadata": {"kind": "procedural_walk", "contacts_preserved": True},
        "frames": frames,
    }


def sample_mapping() -> dict:
    joints = [
        ("pelvis", "pelvis", True),
        ("spine", "spine", True),
        ("left_upleg", "left_upleg", True),
        ("right_upleg", "right_upleg", True),
        ("left_leg", "left_leg", True),
        ("right_leg", "right_leg", True),
        ("left_foot", "left_foot", True),
        ("right_foot", "right_foot", True),
        ("left_toe", "left_toe", True),
        ("right_toe", "right_toe", True),
        ("source_ponytail", None, False),
        (None, "target_tail", False),
    ]
    return {
        "name": "biped_source_a_to_target_b",
        "source_skeleton": "biped_source_a",
        "target_skeleton": "biped_target_b",
        "scale_mode": "leg_length",
        "joint_pairs": [
            {"source": source, "target": target, "required": required}
            for source, target, required in joints
        ],
    }


def sample_batch() -> dict:
    return {
        "name": "sample_walk_transfer",
        "description": "Transfer the supplied walk from source biped A to longer-legged target biped B.",
        "jobs": [
            {
                "source_skeleton": "resources/skeletons/source_a.json",
                "target_skeleton": "resources/skeletons/target_b.json",
                "mapping": "mappings/biped_a_to_b.json",
                "source_animation": "resources/animations/source_walk.json",
                "output_animation": "resources/animations/target_walk_retargeted.json",
            }
        ],
    }
