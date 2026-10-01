from __future__ import annotations

from math import pi
from typing import Any

from .math3d import Vec, quat_to_euler_xyz, vlength, vsub
from .models import Mapping, Motion, Skeleton
from .retarget import mapped_skeleton_scale, pose_world_positions, skeleton_height, source_to_target_position


def check_joint_limits(skeleton: Skeleton, motion: Motion) -> list[dict[str, Any]]:
    failures = []
    for frame_index, frame in enumerate(motion.frames):
        for joint_name, rotation in frame["rotations"].items():
            joint = skeleton.joints.get(joint_name)
            if joint is None or not joint.rotation_limits:
                continue
            euler = quat_to_euler_xyz(tuple(rotation))
            for axis, index in (("x", 0), ("y", 1), ("z", 2)):
                if axis not in joint.rotation_limits:
                    continue
                low, high = joint.rotation_limits[axis]
                value = euler[index]
                if value < low - 1e-6 or value > high + 1e-6:
                    failures.append({
                        "type": "joint_limit",
                        "frame": frame_index,
                        "joint": joint_name,
                        "axis": axis,
                        "value": value,
                        "range": [low, high],
                    })
    return failures


def check_feet(
    source: Skeleton,
    target: Skeleton,
    source_motion: Motion,
    target_motion: Motion,
    mapping: Mapping,
    tolerance: float = 0.25,
) -> list[dict[str, Any]]:
    pairs = mapping.required_pairs()
    feet = {source_name: target_name for source_name, target_name in pairs.items() if source_name.endswith("_foot")}
    if not feet:
        return [{"type": "foot_check", "message": "mapping contains no source *_foot joint"}]

    failures = []
    scale = mapped_skeleton_scale(source, target, pairs)
    for frame_index, (source_frame, target_frame) in enumerate(zip(source_motion.frames, target_motion.frames)):
        source_positions = pose_world_positions(source, source_frame)
        target_positions = pose_world_positions(target, target_frame)
        source_origin = tuple(source_frame["root_translation"])
        target_origin = tuple(target_frame["root_translation"])
        for source_foot, target_foot in feet.items():
            expected = source_to_target_position(
                source_positions[source_foot], source_origin, target_origin, scale
            )
            actual = target_positions[target_foot]
            error = vlength(vsub(expected, actual))
            relative_error = error / max(skeleton_height(target), 1e-9)
            if relative_error > tolerance:
                failures.append({
                    "type": "foot_position",
                    "frame": frame_index,
                    "joint": target_foot,
                    "error": error,
                    "relative_error": relative_error,
                    "expected": expected,
                    "actual": actual,
                })
    return failures


def validate_retarget(
    source: Skeleton,
    target: Skeleton,
    source_motion: Motion,
    target_motion: Motion,
    mapping: Mapping,
) -> dict[str, Any]:
    limit_failures = check_joint_limits(target, target_motion)
    foot_failures = check_feet(source, target, source_motion, target_motion, mapping)
    failures = limit_failures + foot_failures
    return {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "metrics": {
            "frames": len(target_motion.frames),
            "joint_limit_violations": len(limit_failures),
            "foot_position_violations": len(foot_failures),
        },
    }


def expected_contact_position(
    contact: dict[str, Any], frame_index: int, scale: float = 1.0
) -> tuple[float, float, float] | None:
    start = int(contact["start"])
    end = int(contact["end"])
    if not start <= frame_index <= end:
        return None
    blend_in = int(contact.get("blend_in", 0))
    blend_out = int(contact.get("blend_out", 0))
    if blend_in and frame_index < start + blend_in:
        return None
    if blend_out and frame_index > end - blend_out:
        return None
    keyframes = contact.get("target_keyframes") or []
    if keyframes:
        keys = sorted(
            ((int(key["frame"]), tuple(float(v) for v in key["position"])) for key in keyframes),
            key=lambda item: item[0],
        )
        if frame_index <= keys[0][0]:
            return keys[0][1]
        if frame_index >= keys[-1][0]:
            return keys[-1][1]
        for (left_frame, left_pos), (right_frame, right_pos) in zip(keys, keys[1:]):
            if left_frame <= frame_index <= right_frame:
                span = right_frame - left_frame
                t = 0.0 if span == 0 else (frame_index - left_frame) / span
                return tuple(left_pos[i] + (right_pos[i] - left_pos[i]) * t for i in range(3))
    anchor = contact.get("anchor")
    if anchor is not None:
        return tuple(float(value) for value in anchor)
    return None


def check_contact_sliding(
    target: Skeleton,
    motion: Motion,
    tolerance: float = 1e-4,
    conceded_frames: set[tuple[str, int]] | None = None,
) -> list[dict[str, Any]]:
    """Fully-weighted contact frames must hold their (possibly moving) target.

    Entry/exit blend frames are transitional and are not judged as slides;
    frames outside declared intervals keep playing the original motion and
    are never constrained.
    """
    contacts = motion.contacts
    failures: list[dict[str, Any]] = []
    conceded_frames = conceded_frames or set()
    for contact in contacts:
        joint = contact["joint"]
        for frame_index in range(int(contact["start"]), int(contact["end"]) + 1):
            expected = expected_contact_position(contact, frame_index)
            if expected is None:
                continue
            if (joint, frame_index) in conceded_frames:
                continue
            positions = pose_world_positions(target, motion.frames[frame_index])
            position = positions[joint]
            slide = vlength(vsub(position, expected))
            if slide > tolerance:
                failures.append({
                    "type": "contact_slide",
                    "frame": frame_index,
                    "joint": joint,
                    "slide": slide,
                    "expected": list(expected),
                    "actual": list(position),
                })
    return failures


def validate_grounded(
    target: Skeleton,
    motion: Motion,
    grounding_report: dict[str, Any],
    *,
    slide_tolerance: float = 1e-4,
) -> dict[str, Any]:
    slide_failures: list[dict[str, Any]] = []
    unmet = grounding_report.get("unmet", [])
    concessions = grounding_report.get("concessions", [])
    conceded_frames = {
        (item["joint"], int(item["frame"]))
        for item in unmet
        if item.get("conceded_to")
    }
    slide_failures = check_contact_sliding(target, motion, slide_tolerance, conceded_frames)
    return {
        "passed": not slide_failures,
        "failure_count": len(slide_failures),
        "failures": slide_failures,
        "warnings": unmet,
        "warning_count": len(unmet),
        "concessions": concessions,
        "concession_count": len(concessions),
        "metrics": {
            "contact_intervals": len(grounding_report.get("contacts", [])),
            "max_contact_error": grounding_report.get("max_contact_error", 0.0),
            "unmet_contact_frames": len(unmet),
            "conceded_contact_frames": len(concessions),
            "contact_slide_violations": len(slide_failures),
        },
    }
