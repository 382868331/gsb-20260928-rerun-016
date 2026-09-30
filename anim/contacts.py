from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .math3d import (
    Quat,
    Vec,
    q_from_to_directions,
    q_rotate_vector,
    qidentity,
    qinverse,
    qmul,
    quat_from_euler_xyz,
    quat_to_euler_xyz,
    vadd,
    vdot,
    vlength,
    vscale,
    vsub,
)
from .models import Skeleton
from .retarget import local_frame_rotations, pose_world_positions


@dataclass(frozen=True)
class ContactInterval:
    joint: str
    start: int
    end: int
    ground_y: float = 0.0

    def frames(self) -> range:
        return range(self.start, self.end + 1)


def parse_contacts(
    raw_contacts: list[dict[str, Any]], frame_count: int, joints: set[str]
) -> list[ContactInterval]:
    intervals: list[ContactInterval] = []
    for index, raw in enumerate(raw_contacts):
        interval = ContactInterval(
            joint=raw["joint"],
            start=int(raw["start"]),
            end=int(raw["end"]),
            ground_y=float(raw.get("ground_y", 0.0)),
        )
        if interval.joint not in joints:
            raise ValueError(f"contact {index} references unknown foot joint {interval.joint}")
        if interval.start < 0 or interval.end >= frame_count or interval.start > interval.end:
            raise ValueError(f"contact {index} for {interval.joint} has invalid frame range")
        intervals.append(interval)
    return intervals


def _ordered_frame_rotations(skeleton: Skeleton, frame: dict[str, Any]) -> dict[str, Quat]:
    rotations = frame["rotations"]
    result: dict[str, Quat] = {}
    for name in skeleton.ordered():
        joint = skeleton.joints[name]
        local = tuple(rotations.get(name, qidentity()))
        bind = joint.bind_rotation if joint.bind_rotation is not None else qidentity()
        relative = qmul(bind, local)
        result[name] = relative if joint.parent is None else qmul(result[joint.parent], relative)
    return result


def _clamp_limited_joint(
    skeleton: Skeleton, joint_name: str, global_rotation: Quat, parent_global: Quat
) -> tuple[Quat, dict[str, Any] | None]:
    joint = skeleton.joints[joint_name]
    bind = joint.bind_rotation if joint.bind_rotation is not None else qidentity()
    local = qmul(qinverse(bind), qmul(qinverse(parent_global), global_rotation))
    if not joint.rotation_limits:
        return global_rotation, None
    euler = list(quat_to_euler_xyz(local))
    clipped: dict[str, Any] = {}
    for axis, axis_index in (("x", 0), ("y", 1), ("z", 2)):
        if axis not in joint.rotation_limits:
            continue
        low, high = joint.rotation_limits[axis]
        value = euler[axis_index]
        bounded = min(max(value, low), high)
        if abs(bounded - value) > 1e-6:
            clipped[axis] = {"value": value, "clamped": bounded, "range": [low, high]}
        euler[axis_index] = bounded
    if not clipped:
        return global_rotation, None
    clamped_local = quat_from_euler_xyz(tuple(euler))
    clamped_global = qmul(parent_global, qmul(bind, clamped_local))
    return clamped_global, {"joint": joint_name, "axes": clipped}


def _leg_chain(skeleton: Skeleton, foot: str) -> tuple[str, str]:
    knee = skeleton.joints[foot].parent
    if knee is None:
        raise ValueError(f"foot joint {foot} has no parent chain")
    hip = skeleton.joints[knee].parent
    if hip is None:
        raise ValueError(f"knee joint {knee} has no parent")
    return hip, knee


def _fk_update(
    skeleton: Skeleton,
    globals_: dict[str, Quat],
    positions: dict[str, Vec],
    chain: tuple[str, str],
    foot: str,
) -> None:
    hip, knee = chain
    for name, parent_name in ((knee, hip), (foot, knee)):
        positions[name] = tuple(
            positions[parent_name][i] + q_rotate_vector(globals_[parent_name], skeleton.joints[name].offset)[i]
            for i in range(3)
        )


def _apply_two_bone_ik(
    skeleton: Skeleton,
    globals_: dict[str, Quat],
    positions: dict[str, Vec],
    foot: str,
    target: Vec,
) -> tuple[dict[str, Quat], Vec, list[dict[str, Any]]]:
    hip_name, knee_name = _leg_chain(skeleton, foot)
    upper_length = vlength(skeleton.joints[knee_name].offset)
    lower_length = vlength(skeleton.joints[foot].offset)
    hip_pos = positions[hip_name]
    to_target = vsub(target, hip_pos)
    distance = vlength(to_target)
    limit_conflicts: list[dict[str, Any]] = []

    if distance < 1e-10:
        direction = (0.0, -1.0, 0.0)
        reach = upper_length + lower_length
    else:
        direction = vscale(to_target, 1.0 / distance)
        reach = min(max(distance, abs(upper_length - lower_length) + 1e-6), upper_length + lower_length - 1e-6)

    extension = (upper_length * upper_length - lower_length * lower_length + reach * reach) / (2.0 * reach)
    extension = min(max(extension, 0.0), upper_length)
    center = vadd(hip_pos, vscale(direction, extension))
    radius = (max(upper_length * upper_length - extension * extension, 0.0)) ** 0.5

    current_knee = positions[knee_name]
    along = vdot(vsub(current_knee, hip_pos), direction)
    pole = vsub(current_knee, vadd(hip_pos, vscale(direction, along)))
    if vlength(pole) < 1e-8:
        pole = q_rotate_vector(globals_[hip_name], skeleton.joints[knee_name].offset)
        pole = vsub(pole, vscale(direction, vdot(pole, direction)))
    if vlength(pole) < 1e-8:
        world_up = (0.0, 1.0, 0.0)
        pole = vsub(world_up, vscale(direction, vdot(world_up, direction)))
    if vlength(pole) < 1e-8:
        pole = (0.0, 0.0, 1.0)
    pole = vscale(pole, 1.0 / vlength(pole))
    knee_target = vadd(center, vscale(pole, radius))
    foot_reached = vadd(hip_pos, vscale(direction, reach))

    hip_parent = (
        qidentity()
        if skeleton.joints[hip_name].parent is None
        else globals_[skeleton.joints[hip_name].parent]
    )
    upper_current = q_rotate_vector(globals_[hip_name], skeleton.joints[knee_name].offset)
    hip_global = qmul(globals_[hip_name], q_from_to_directions(upper_current, vsub(knee_target, hip_pos)))
    hip_global, hip_limit = _clamp_limited_joint(skeleton, hip_name, hip_global, hip_parent)
    if hip_limit:
        limit_conflicts.append(hip_limit)

    actual_knee = tuple(
        hip_pos[i] + q_rotate_vector(hip_global, skeleton.joints[knee_name].offset)[i] for i in range(3)
    )
    lower_current = q_rotate_vector(globals_[knee_name], skeleton.joints[foot].offset)
    local_correction = q_from_to_directions(lower_current, vsub(foot_reached, actual_knee))
    old_hip_global = globals_[hip_name]
    knee_local = qmul(qinverse(old_hip_global), globals_[knee_name])
    knee_global = qmul(hip_global, qmul(local_correction, knee_local))
    knee_global, knee_limit = _clamp_limited_joint(skeleton, knee_name, knee_global, hip_global)
    if knee_limit:
        limit_conflicts.append(knee_limit)

    updated = dict(globals_)
    updated[hip_name] = hip_global
    updated[knee_name] = knee_global
    updated_positions = dict(positions)
    updated_positions[knee_name] = actual_knee
    updated_positions[foot] = tuple(
        actual_knee[i] + q_rotate_vector(knee_global, skeleton.joints[foot].offset)[i] for i in range(3)
    )
    residual = vsub(updated_positions[foot], target)
    for _ in range(3):
        if vlength(residual) <= 1e-7:
            break
        shin_refine = q_rotate_vector(knee_global, skeleton.joints[foot].offset)
        knee_refine = qmul(
            hip_global, qmul(q_from_to_directions(shin_refine, vsub(target, actual_knee)), qmul(qinverse(hip_global), knee_global))
        )
        knee_global, second_knee_limit = _clamp_limited_joint(
            skeleton, knee_name, knee_refine, hip_global
        )
        if second_knee_limit:
            limit_conflicts.append(second_knee_limit)
        updated[knee_name] = knee_global
        updated_positions[foot] = tuple(
            actual_knee[i] + q_rotate_vector(knee_global, skeleton.joints[foot].offset)[i] for i in range(3)
        )
        residual = vsub(updated_positions[foot], target)
    return updated, residual, limit_conflicts


def _horizontal_root_correction(
    skeleton: Skeleton,
    positions: dict[str, Vec],
    foot: str,
    anchor: Vec,
) -> Vec:
    """Move the root horizontally (height preserved) so the anchor is reachable."""
    hip, knee = _leg_chain(skeleton, foot)
    max_reach = vlength(skeleton.joints[knee].offset) + vlength(skeleton.joints[foot].offset) - 1e-6
    horizontal_hip = (anchor[0] - positions[hip][0], anchor[2] - positions[hip][2])
    horizontal_distance = (horizontal_hip[0] ** 2 + horizontal_hip[1] ** 2) ** 0.5
    three_dimensional_distance = vlength(vsub(anchor, positions[hip]))
    correction = [0.0, 0.0, 0.0]
    vertical_gap = anchor[1] - positions[hip][1]
    if three_dimensional_distance > max_reach:
        abs_vertical = abs(vertical_gap)
        max_horizontal_squared = max_reach ** 2 - abs_vertical ** 2
        if max_horizontal_squared < 0.0:
            magnitude = abs_vertical - max_reach
            correction[1] = magnitude if vertical_gap > 0.0 else -magnitude
        elif horizontal_distance > max_horizontal_squared ** 0.5:
            excess = horizontal_distance - max_horizontal_squared ** 0.5
            direction = (horizontal_hip[0] / max(horizontal_distance, 1e-12), horizontal_hip[1] / max(horizontal_distance, 1e-12))
            correction[0] = direction[0] * excess
            correction[2] = direction[1] * excess
    return tuple(correction)  # type: ignore[return-value]


def apply_foot_contacts(
    target: Skeleton,
    motion_data: dict[str, Any],
    *,
    ground_scale: float = 1.0,
) -> dict[str, Any]:
    """Pin feet during declared contact intervals; swing frames stay untouched.

    When configured joint limits keep the foot from reaching the anchor, the
    leg is clamped to the limits and the remaining offset is reported as an
    interpretable unmet contact instead of forcing the pose out of range.
    """
    frames = motion_data["frames"]
    feet = {joint.name for joint in target.joints.values() if joint.name.endswith("_foot")}
    intervals = parse_contacts(motion_data.get("contacts", []), len(frames), feet)
    if not intervals:
        return {"contacts": [], "unmet": [], "max_contact_error": 0.0}

    unmet: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    max_error = 0.0

    for interval in intervals:
        foot = interval.joint
        chain = _leg_chain(target, foot)
        start_foot = pose_world_positions(target, frames[interval.start])[foot]
        ground_y = interval.ground_y * ground_scale
        anchor: Vec = (start_foot[0], ground_y, start_foot[2])
        interval_max = 0.0
        unmet_frames: list[int] = []

        for frame_index in interval.frames():
            frame = frames[frame_index]
            root_adjustment_total = (0.0, 0.0, 0.0)
            residual: Vec = (0.0, 0.0, 0.0)
            limit_conflicts: list[dict[str, Any]] = []
            for iteration in range(6):
                positions = pose_world_positions(target, frame)
                correction = _horizontal_root_correction(target, positions, foot, anchor)
                if vlength(correction) > 1e-9:
                    root = frame["root_translation"]
                    frame["root_translation"] = [root[i] + correction[i] for i in range(3)]
                    root_adjustment_total = tuple(root_adjustment_total[i] + correction[i] for i in range(3))
                globals_ = _ordered_frame_rotations(target, frame)
                positions = pose_world_positions(target, frame)
                globals_, residual, limit_conflicts = _apply_two_bone_ik(
                    target, globals_, positions, foot, anchor
                )
                locals_ = local_frame_rotations(target, globals_)
                for name in (*chain, foot):
                    frame["rotations"][name] = [float(value) for value in locals_[name]]
                if vlength(correction) < 1e-8:
                    break

            error = vlength(residual)
            interval_max = max(interval_max, error)
            max_error = max(max_error, error)
            if error > 1e-5 or limit_conflicts:
                unmet_frames.append(frame_index)
                unmet.append({
                    "type": "unmet_contact",
                    "frame": frame_index,
                    "joint": foot,
                    "anchor": list(anchor),
                    "residual": list(residual),
                    "error": error,
                    "root_adjustment": list(root_adjustment_total),
                    "limit_conflicts": limit_conflicts,
                    "reason": "outside leg reach" if error > 1e-5 else "joint limit conflict",
                })

        reports.append({
            "joint": foot,
            "start": interval.start,
            "end": interval.end,
            "ground_y": ground_y,
            "ground_scale": ground_scale,
            "anchor": list(anchor),
            "max_error": interval_max,
            "unmet_frames": unmet_frames,
        })

    motion_data["grounding"] = {
        "policy": "contact feet pinned to per-interval anchors via two-bone IK; "
        "limit conflicts are clamped and the residual is reported",
        "intervals": reports,
    }
    return {"contacts": reports, "unmet": unmet, "max_contact_error": max_error}
