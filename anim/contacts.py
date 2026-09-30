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
    vlerp,
    vlength,
    vscale,
    vsub,
)
from .models import Skeleton
from .retarget import local_frame_rotations, pose_world_positions


LOCK_WEIGHT = 1.0 - 1e-9


@dataclass(frozen=True)
class ContactInterval:
    joint: str
    start: int
    end: int
    ground_y: float = 0.0
    priority: int = 0
    blend_in: int = 0
    blend_out: int = 0
    target_path: tuple[tuple[float, float, float], ...] | None = None
    velocity: tuple[float, float, float] | None = None
    name: str | None = None

    def frames(self) -> range:
        return range(self.start, self.end + 1)

    @property
    def moving(self) -> bool:
        return self.target_path is not None or self.velocity is not None


def _smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def contact_weight(contact: ContactInterval, frame_index: int) -> float:
    """Smooth 0->1 ramp over blend_in frames and 1->0 over blend_out frames."""
    if frame_index < contact.start or frame_index > contact.end:
        return 0.0
    weight = 1.0
    if contact.blend_in > 0:
        weight = min(weight, _smoothstep((frame_index - contact.start + 1) / (contact.blend_in + 1)))
    if contact.blend_out > 0:
        weight = min(weight, _smoothstep((contact.end - frame_index + 1) / (contact.blend_out + 1)))
    return weight


def contact_weight_from_raw(raw: dict[str, Any], frame_index: int) -> float:
    interval = ContactInterval(
        joint=raw["joint"],
        start=int(raw["start"]),
        end=int(raw["end"]),
        blend_in=int(raw.get("blend_in", 0)),
        blend_out=int(raw.get("blend_out", 0)),
    )
    return contact_weight(interval, frame_index)


def parse_contacts(
    raw_contacts: list[dict[str, Any]], frame_count: int, joints: set[str]
) -> list[ContactInterval]:
    intervals: list[ContactInterval] = []
    for index, raw in enumerate(raw_contacts):
        target_path = raw.get("target_path")
        velocity = raw.get("velocity")
        interval = ContactInterval(
            joint=raw["joint"],
            start=int(raw["start"]),
            end=int(raw["end"]),
            ground_y=float(raw.get("ground_y", 0.0)),
            priority=int(raw.get("priority", 0)),
            blend_in=int(raw.get("blend_in", 0)),
            blend_out=int(raw.get("blend_out", 0)),
            target_path=(
                tuple(tuple(float(c) for c in point) for point in target_path)
                if target_path is not None
                else None
            ),
            velocity=tuple(float(c) for c in velocity) if velocity is not None else None,
            name=raw.get("name"),
        )
        if interval.joint not in joints:
            raise ValueError(f"contact {index} references unknown or chain-less joint {interval.joint}")
        if interval.start < 0 or interval.end >= frame_count or interval.start > interval.end:
            raise ValueError(f"contact {index} for {interval.joint} has invalid frame range")
        if interval.target_path is not None and interval.velocity is not None:
            raise ValueError(f"contact {index} for {interval.joint} cannot combine target_path and velocity")
        if interval.target_path is not None and len(interval.target_path) != interval.end - interval.start + 1:
            raise ValueError(
                f"contact {index} for {interval.joint} target_path must cover every frame of the interval"
            )
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


def _two_bone_chain(skeleton: Skeleton, joint_name: str) -> tuple[str, str]:
    """Return the (grandparent, parent) chain driving an end joint (hip/knee or shoulder/elbow)."""
    knee = skeleton.joints[joint_name].parent
    if knee is None:
        raise ValueError(f"joint {joint_name} has no parent chain")
    hip = skeleton.joints[knee].parent
    if hip is None:
        raise ValueError(f"knee joint {knee} has no parent")
    return hip, knee


def _chainable_joints(skeleton: Skeleton) -> set[str]:
    result = set()
    for name, joint in skeleton.joints.items():
        if joint.parent is not None and skeleton.joints[joint.parent].parent is not None:
            result.add(name)
    return result


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
    pole_flip: bool = False,
) -> tuple[dict[str, Quat], Vec, list[dict[str, Any]]]:
    hip_name, knee_name = _two_bone_chain(skeleton, foot)
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
    if pole_flip:
        pole = vscale(pole, -1.0)
    knee_target = vadd(center, vscale(pole, radius))
    foot_reached = vadd(hip_pos, vscale(direction, reach))

    hip_parent = (
        qidentity()
        if skeleton.joints[hip_name].parent is None
        else globals_[skeleton.joints[hip_name].parent]
    )
    upper_current = q_rotate_vector(globals_[hip_name], skeleton.joints[knee_name].offset)
    hip_global = qmul(
        q_from_to_directions(vsub(knee_target, hip_pos), upper_current), globals_[hip_name]
    )
    hip_global, hip_limit = _clamp_limited_joint(skeleton, hip_name, hip_global, hip_parent)
    if hip_limit:
        limit_conflicts.append(hip_limit)

    actual_knee = tuple(
        hip_pos[i] + q_rotate_vector(hip_global, skeleton.joints[knee_name].offset)[i] for i in range(3)
    )
    lower_current = q_rotate_vector(globals_[knee_name], skeleton.joints[foot].offset)
    world_correction = q_from_to_directions(vsub(foot_reached, actual_knee), lower_current)
    knee_global = qmul(world_correction, globals_[knee_name])
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
        knee_refine = qmul(q_from_to_directions(vsub(target, actual_knee), shin_refine), knee_global)
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


def _conflict_score(conflicts: list[dict[str, Any]]) -> float:
    score = 0.0
    for conflict in conflicts:
        for axis in conflict["axes"].values():
            score += abs(axis["value"] - axis["clamped"])
    return score


def _solve_two_bone_ik(
    skeleton: Skeleton,
    globals_: dict[str, Quat],
    positions: dict[str, Vec],
    foot: str,
    target: Vec,
) -> tuple[dict[str, Quat], Vec, list[dict[str, Any]]]:
    """Solve two-bone IK, retrying with the opposite knee/elbow pole side when
    the first solution has to be clamped by joint limits."""
    updated, residual, conflicts = _apply_two_bone_ik(
        skeleton, globals_, positions, foot, target
    )
    if not conflicts:
        return updated, residual, conflicts
    flipped, flipped_residual, flipped_conflicts = _apply_two_bone_ik(
        skeleton, globals_, positions, foot, target, pole_flip=True
    )
    current_score = _conflict_score(conflicts) + vlength(residual)
    flipped_score = _conflict_score(flipped_conflicts) + vlength(flipped_residual)
    if flipped_score < current_score:
        return flipped, flipped_residual, flipped_conflicts
    return updated, residual, conflicts


def _horizontal_root_correction(
    skeleton: Skeleton,
    positions: dict[str, Vec],
    foot: str,
    anchor: Vec,
) -> Vec:
    """Move the root horizontally (height preserved) so the anchor is reachable."""
    hip, knee = _two_bone_chain(skeleton, foot)
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
    """Solve declared contacts on the target skeleton; free frames stay untouched.

    Multiple contacts may overlap in time (both feet plus a hand on a rail).
    Contacts are solved per frame in priority order; when targets conflict,
    the lower-priority contact keeps its residual error and is reported as
    yielded instead of being marked successful.  Root translation corrections
    only serve the highest-priority tier of active contacts, so a low-priority
    hand hold can never drag the body off its feet.  Anchors may move with a
    support (per-frame target_path or constant velocity), and blend_in /
    blend_out frames ramp each contact in and out smoothly.  When configured
    joint limits keep a joint from reaching its anchor, the chain is clamped
    to the limits and the remaining offset is reported as an interpretable
    unmet contact instead of forcing the pose out of range.
    """
    frames = motion_data["frames"]
    frame_rate = float(motion_data.get("frame_rate", 30.0))
    chainable = _chainable_joints(target)
    intervals = parse_contacts(motion_data.get("contacts", []), len(frames), chainable)
    if not intervals:
        return {"contacts": [], "unmet": [], "max_contact_error": 0.0}

    unmet: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    max_error = 0.0

    start_anchors: dict[int, Vec] = {}
    scaled_paths: dict[int, tuple[Vec, ...] | None] = {}
    scaled_velocities: dict[int, Vec | None] = {}
    for index, interval in enumerate(intervals):
        start_position = pose_world_positions(target, frames[interval.start])[interval.joint]
        ground_y = interval.ground_y * ground_scale
        start_anchors[index] = (start_position[0], ground_y, start_position[2])
        if interval.target_path is not None:
            scaled_paths[index] = tuple(
                (point[0] * ground_scale, point[1] * ground_scale, point[2] * ground_scale)
                for point in interval.target_path
            )
        else:
            scaled_paths[index] = None
        scaled_velocities[index] = (
            vscale(interval.velocity, ground_scale) if interval.velocity is not None else None
        )

    def anchor_at(index: int, frame_index: int) -> Vec:
        interval = intervals[index]
        offset = frame_index - interval.start
        if scaled_paths[index] is not None:
            return scaled_paths[index][offset]
        if scaled_velocities[index] is not None:
            dt = offset / max(frame_rate, 1e-9)
            return vadd(start_anchors[index], vscale(scaled_velocities[index], dt))
        return start_anchors[index]

    interval_errors: dict[int, float] = {index: 0.0 for index in range(len(intervals))}
    interval_unmet_frames: dict[int, list[int]] = {index: [] for index in range(len(intervals))}
    interval_root_adjust: dict[int, float] = {index: 0.0 for index in range(len(intervals))}

    for frame_index, frame in enumerate(frames):
        active = [
            (index, contact_weight(intervals[index], frame_index))
            for index in range(len(intervals))
            if intervals[index].start <= frame_index <= intervals[index].end
        ]
        active = [(index, weight) for index, weight in active if weight > 1e-9]
        if not active:
            continue
        active.sort(key=lambda item: (intervals[item[0]].priority, intervals[item[0]].joint))

        original_positions = pose_world_positions(target, frame)
        root_adjustment_total: Vec = (0.0, 0.0, 0.0)
        limit_conflicts: dict[int, list[dict[str, Any]]] = {index: [] for index, _ in active}
        root_priority = min(intervals[index].priority for index, _ in active)

        for _round in range(6):
            largest_correction = 0.0
            for index, weight in active:
                interval = intervals[index]
                anchor = anchor_at(index, frame_index)
                blended_target = vlerp(original_positions[interval.joint], anchor, weight)
                if intervals[index].priority == root_priority:
                    positions = pose_world_positions(target, frame)
                    correction = _horizontal_root_correction(
                        target, positions, interval.joint, blended_target
                    )
                    correction = vscale(correction, weight)
                    if vlength(correction) > 1e-9:
                        root = frame["root_translation"]
                        frame["root_translation"] = [root[i] + correction[i] for i in range(3)]
                        root_adjustment_total = vadd(root_adjustment_total, correction)
                        largest_correction = max(largest_correction, vlength(correction))
                globals_ = _ordered_frame_rotations(target, frame)
                positions = pose_world_positions(target, frame)
                globals_, _residual, conflicts = _solve_two_bone_ik(
                    target, globals_, positions, interval.joint, blended_target
                )
                limit_conflicts[index] = conflicts
                locals_ = local_frame_rotations(target, globals_)
                chain = _two_bone_chain(target, interval.joint)
                for name in (*chain, interval.joint):
                    frame["rotations"][name] = [float(value) for value in locals_[name]]
            if largest_correction < 1e-8:
                break

        final_positions = pose_world_positions(target, frame)
        frame_errors: dict[int, float] = {}
        for index, weight in active:
            interval = intervals[index]
            anchor = anchor_at(index, frame_index)
            residual = vsub(final_positions[interval.joint], anchor)
            error = vlength(residual)
            frame_errors[index] = error
            interval_root_adjust[index] = max(
                interval_root_adjust[index], vlength(root_adjustment_total)
            )
            if weight < LOCK_WEIGHT:
                continue
            interval_errors[index] = max(interval_errors[index], error)
            max_error = max(max_error, error)
            if error > 1e-5 or limit_conflicts[index]:
                interval_unmet_frames[index].append(frame_index)
                higher_met = [
                    intervals[other].joint
                    for other, other_weight in active
                    if other_weight >= LOCK_WEIGHT
                    and intervals[other].priority < interval.priority
                    and frame_errors.get(other, 0.0) <= 1e-5
                ]
                unmet.append({
                    "type": "unmet_contact",
                    "frame": frame_index,
                    "joint": interval.joint,
                    "contact": interval.name,
                    "priority": interval.priority,
                    "anchor": list(anchor),
                    "residual": list(residual),
                    "error": error,
                    "root_adjustment": list(root_adjustment_total),
                    "limit_conflicts": limit_conflicts[index],
                    "yielded": bool(higher_met),
                    "yielded_to": higher_met,
                    "reason": (
                        "yielded to higher priority contact"
                        if higher_met
                        else ("outside reach" if error > 1e-5 else "joint limit conflict")
                    ),
                })

    for index, interval in enumerate(intervals):
        anchor_path = [list(anchor_at(index, frame)) for frame in interval.frames()]
        raw_contact = motion_data["contacts"][index]
        raw_contact["anchor_path"] = anchor_path
        raw_contact["priority"] = interval.priority
        if interval.blend_in:
            raw_contact["blend_in"] = interval.blend_in
        if interval.blend_out:
            raw_contact["blend_out"] = interval.blend_out
        reports.append({
            "joint": interval.joint,
            "contact": interval.name,
            "start": interval.start,
            "end": interval.end,
            "ground_y": interval.ground_y * ground_scale,
            "ground_scale": ground_scale,
            "priority": interval.priority,
            "blend_in": interval.blend_in,
            "blend_out": interval.blend_out,
            "moving": interval.moving,
            "anchor": list(start_anchors[index]),
            "anchor_path": anchor_path,
            "max_error": interval_errors[index],
            "max_root_adjustment": interval_root_adjust[index],
            "unmet_frames": interval_unmet_frames[index],
            "met": not interval_unmet_frames[index],
        })

    motion_data["grounding"] = {
        "policy": "contacts solved per frame in priority order via two-bone IK; "
        "overlapping targets that conflict yield to higher priority contacts and "
        "keep their residual error; moving supports are tracked through "
        "target_path/velocity anchors; blend_in/blend_out frames ramp contacts "
        "smoothly; limit conflicts are clamped and the residual is reported",
        "intervals": reports,
    }
    return {"contacts": reports, "unmet": unmet, "max_contact_error": max_error}
