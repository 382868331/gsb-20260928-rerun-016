from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .fk import forward_kinematics, rest_pose
from .math3d import euler_xyz
from .models import Animation, Skeleton


@dataclass
class ValidationReport:
    passed: bool
    checks: list[dict]
    errors: list[str]

    def to_dict(self) -> dict:
        return {"passed": self.passed, "checks": self.checks, "errors": self.errors}


def _contact_frames(positions_by_frame: list[dict[str, np.ndarray]], joint: str, frame_time: float) -> list[int]:
    heights = np.array([positions[joint][1] for positions in positions_by_frame])
    speeds = np.zeros(len(heights))
    for i in range(1, len(heights)):
        speeds[i] = np.linalg.norm(positions_by_frame[i][joint] - positions_by_frame[i - 1][joint]) / frame_time
    ground_height = float(np.min(heights))
    return [i for i, _ in enumerate(heights) if heights[i] <= ground_height + 0.03 and speeds[i] <= 0.12]


def validate_animation(
    skeleton: Skeleton,
    animation: Animation,
    *,
    foot_slide_tolerance: float = 0.025,
    joint_limit_tolerance: float = 1e-5,
    source_animation: Animation | None = None,
    source_skeleton: Skeleton | None = None,
    scale: float = 1.0,
) -> ValidationReport:
    errors: list[str] = []
    checks: list[dict] = []
    if animation.skeleton_name != skeleton.name:
        errors.append("animation skeleton does not match validation skeleton")
        return ValidationReport(False, checks, errors)

    target_frames = [
        forward_kinematics(skeleton, animation.local_rotations[i], animation.root_translations[i])
        for i in range(animation.frame_count)
    ]
    target_positions = [positions for positions, _ in target_frames]

    root_path = 0.0
    for previous, current in zip(animation.root_translations, animation.root_translations[1:]):
        root_path += float(np.linalg.norm(current - previous))
    checks.append({"name": "root_motion_distance", "value": root_path, "unit": "m"})
    if source_animation is not None and source_skeleton is not None:
        source_root_path = sum(
            float(np.linalg.norm(b - a))
            for a, b in zip(source_animation.root_translations, source_animation.root_translations[1:])
        )
        expected_horizontal = source_root_path * scale
        if root_path + 1e-6 < expected_horizontal * 0.5:
            errors.append("root motion was not transferred to target skeleton")

    limit_violations: list[dict] = []
    for frame_index, rotations in enumerate(animation.local_rotations):
        for name, joint in skeleton.joints.items():
            if joint.limits_rad is None or name not in rotations:
                continue
            angles = euler_xyz(rotations[name])
            for axis_index, (minimum, maximum) in enumerate(joint.limits_rad):
                if angles[axis_index] < minimum - joint_limit_tolerance or angles[axis_index] > maximum + joint_limit_tolerance:
                    limit_violations.append(
                        {
                            "frame": frame_index,
                            "joint": name,
                            "axis": "xyz"[axis_index],
                            "value_rad": float(angles[axis_index]),
                            "min_rad": minimum,
                            "max_rad": maximum,
                        }
                    )
    checks.append({"name": "joint_limit_violations", "value": len(limit_violations)})
    if limit_violations:
        errors.append(f"{len(limit_violations)} joint limit violations")

    if source_animation is not None and source_skeleton is not None:
        source_frames = [
            forward_kinematics(
                source_skeleton,
                source_animation.local_rotations[i],
                source_animation.root_translations[i],
            )
            for i in range(source_animation.frame_count)
        ]
        source_positions = [positions for positions, _ in source_frames]
        source_rest_root, _ = rest_pose(source_skeleton)
        target_rest_root, _ = rest_pose(skeleton)
        foot_errors = []
        for foot_name in ("left_foot", "right_foot"):
            if foot_name not in skeleton.joints or foot_name not in source_skeleton.joints:
                continue
            contacts = _contact_frames(source_positions, foot_name, animation.frame_time)
            max_slide = 0.0
            if contacts:
                anchored = contacts[0]
                source_anchor_offset = source_positions[anchored][foot_name] - source_rest_root[source_skeleton.root_name]
                expected_anchor = target_rest_root[skeleton.root_name] + scale * source_anchor_offset
                for frame in contacts:
                    slide = float(np.linalg.norm(target_positions[frame][foot_name] - expected_anchor))
                    max_slide = max(max_slide, slide)
            for frame in range(animation.frame_count):
                source_offset = source_positions[frame][foot_name] - source_rest_root[source_skeleton.root_name]
                expected = target_rest_root[skeleton.root_name] + scale * source_offset
                error = float(np.linalg.norm(target_positions[frame][foot_name] - expected))
                foot_errors.append(error)
            checks.append(
                {
                    "name": f"{foot_name}_contact_frames",
                    "value": len(contacts),
                    "max_contact_slide_m": max_slide,
                }
            )
            if max_slide > foot_slide_tolerance:
                errors.append(f"{foot_name} slides {max_slide:.4f}m during contact")
        if foot_errors:
            checks.append({"name": "foot_position_max_error", "value": max(foot_errors), "unit": "m"})
            if max(foot_errors) > foot_slide_tolerance:
                errors.append("retargeted feet do not preserve source foot positions")

    return ValidationReport(not errors, checks, errors)
