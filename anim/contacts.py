from __future__ import annotations

from dataclasses import dataclass, field
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

CONTACT_TOLERANCE = 1e-3


@dataclass(frozen=True)
class ContactKeyframe:
    frame: int
    position: Vec


@dataclass(frozen=True)
class ContactInterval:
    joint: str
    start: int
    end: int
    ground_y: float = 0.0
    priority: int = 10
    blend_in: int = 0
    blend_out: int = 0
    keyframes: tuple[ContactKeyframe, ...] = field(default_factory=tuple)
    index: int = 0

    def frames(self) -> range:
        return range(self.start, self.end + 1)

    def weight(self, frame: int) -> float:
        if frame < self.start or frame > self.end:
            return 0.0
        if self.blend_in and frame < self.start + self.blend_in:
            return (frame - self.start + 1) / float(self.blend_in + 1)
        if self.blend_out and frame > self.end - self.blend_out:
            return (self.end - frame + 1) / float(self.blend_out + 1)
        return 1.0

    def target_position(
        self, frame: int, anchor: Vec | None = None, ground_scale: float = 1.0
    ) -> Vec:
        """Interpolated world target for a moving support.

        Keyframe positions are authored in source units and scaled once here;
        the anchor is ignored because the first keyframe fully defines the
        initial world position.
        """
        if not self.keyframes:
            return anchor if anchor is not None else (0.0, 0.0, 0.0)
        keys = sorted(self.keyframes, key=lambda item: item.frame)
        if frame <= keys[0].frame:
            base = keys[0].position
        elif frame >= keys[-1].frame:
            base = keys[-1].position
        else:
            base = keys[-1].position
            for left, right in zip(keys, keys[1:]):
                if left.frame <= frame <= right.frame:
                    span = right.frame - left.frame
                    t = 0.0 if span == 0 else (frame - left.frame) / float(span)
                    base = vlerp(left.position, right.position, t)
                    break
        return tuple(float(value) * ground_scale for value in base)


def parse_contacts(
    raw_contacts: list[dict[str, Any]], frame_count: int, joints: set[str]
) -> list[ContactInterval]:
    intervals: list[ContactInterval] = []
    for index, raw in enumerate(raw_contacts):
        keyframes = tuple(
            ContactKeyframe(
                frame=int(key["frame"]),
                position=tuple(float(value) for value in key["position"]),  # type: ignore[arg-type]
            )
            for key in raw.get("target_keyframes", [])
        )
        interval = ContactInterval(
            joint=raw["joint"],
            start=int(raw["start"]),
            end=int(raw["end"]),
            ground_y=float(raw.get("ground_y", 0.0)),
            priority=int(raw.get("priority", 10)),
            blend_in=max(int(raw.get("blend_in", 0)), 0),
            blend_out=max(int(raw.get("blend_out", 0)), 0),
            keyframes=keyframes,
            index=index,
        )
        if interval.joint not in joints:
            raise ValueError(f"contact {index} references unknown end effector joint {interval.joint}")
        if interval.start < 0 or interval.end >= frame_count or interval.start > interval.end:
            raise ValueError(f"contact {index} for {interval.joint} has invalid frame range")
        span = interval.end - interval.start + 1
        if interval.blend_in + interval.blend_out >= span:
            raise ValueError(
                f"contact {index} for {interval.joint} blend windows exceed interval length"
            )
        for key in keyframes:
            if len(key.position) != 3:
                raise ValueError(f"contact {index} keyframe positions need three components")
            if not interval.start <= key.frame <= interval.end:
                raise ValueError(f"contact {index} keyframe {key.frame} is outside interval")
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


def effector_chain(skeleton: Skeleton, effector: str) -> tuple[str, str]:
    mid = skeleton.joints[effector].parent
    if mid is None:
        raise ValueError(f"end effector {effector} has no parent chain")
    base = skeleton.joints[mid].parent
    if base is None:
        raise ValueError(f"joint {mid} has no parent")
    return base, mid


def _apply_two_bone_ik(
    skeleton: Skeleton,
    globals_: dict[str, Quat],
    positions: dict[str, Vec],
    effector: str,
    target: Vec,
) -> tuple[dict[str, Quat], Vec, list[dict[str, Any]]]:
    base_name, mid_name = effector_chain(skeleton, effector)
    root_parent = skeleton.joints[base_name].parent
    upper_length = vlength(skeleton.joints[mid_name].offset)
    lower_length = vlength(skeleton.joints[effector].offset)
    base_pos = positions[base_name]
    to_target = vsub(target, base_pos)
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
    center = vadd(base_pos, vscale(direction, extension))
    radius = (max(upper_length * upper_length - extension * extension, 0.0)) ** 0.5

    anatomical = (0.0, 0.0, 1.0) if effector.endswith("_foot") else (0.0, -1.0, 0.0)
    preferred = vsub(anatomical, vscale(direction, vdot(anatomical, direction)))
    if vlength(preferred) < 1e-8:
        current_mid = positions[mid_name]
        along = vdot(vsub(current_mid, base_pos), direction)
        preferred = vsub(current_mid, vadd(base_pos, vscale(direction, along)))
    pole = vscale(preferred, 1.0 / max(vlength(preferred), 1e-12))
    mid_target = vadd(center, vscale(pole, radius))
    effector_reached = vadd(base_pos, vscale(direction, reach))

    base_parent = qidentity() if root_parent is None else globals_[root_parent]
    upper_current = q_rotate_vector(globals_[base_name], skeleton.joints[mid_name].offset)
    base_global = qmul(
        q_from_to_directions(upper_current, vsub(mid_target, base_pos)),
        globals_[base_name],
    )
    base_global, base_limit = _clamp_limited_joint(skeleton, base_name, base_global, base_parent)
    if base_limit:
        limit_conflicts.append(base_limit)

    actual_mid = tuple(
        base_pos[i] + q_rotate_vector(base_global, skeleton.joints[mid_name].offset)[i] for i in range(3)
    )
    lower_current = q_rotate_vector(globals_[mid_name], skeleton.joints[effector].offset)
    local_correction = q_from_to_directions(lower_current, vsub(effector_reached, actual_mid))
    corrected_mid_global = qmul(local_correction, globals_[mid_name])
    mid_global = qmul(base_global, qmul(qinverse(globals_[base_name]), corrected_mid_global))
    mid_global, mid_limit = _clamp_limited_joint(skeleton, mid_name, mid_global, base_global)
    if mid_limit:
        limit_conflicts.append(mid_limit)

    updated = dict(globals_)
    updated[base_name] = base_global
    updated[mid_name] = mid_global
    updated_positions = dict(positions)
    updated_positions[mid_name] = actual_mid
    updated_positions[effector] = tuple(
        actual_mid[i] + q_rotate_vector(mid_global, skeleton.joints[effector].offset)[i] for i in range(3)
    )
    residual = vsub(updated_positions[effector], target)
    for _ in range(3):
        if vlength(residual) <= 1e-7:
            break
        forearm_refine = q_rotate_vector(mid_global, skeleton.joints[effector].offset)
        world_correction = q_from_to_directions(forearm_refine, vsub(target, actual_mid))
        corrected_global = qmul(world_correction, mid_global)
        mid_global = qmul(base_global, qmul(qinverse(base_global), corrected_global))
        mid_global, second_mid_limit = _clamp_limited_joint(
            skeleton, mid_name, mid_global, base_global
        )
        if second_mid_limit:
            limit_conflicts.append(second_mid_limit)
        updated[mid_name] = mid_global
        updated_positions[effector] = tuple(
            actual_mid[i] + q_rotate_vector(mid_global, skeleton.joints[effector].offset)[i]
            for i in range(3)
        )
        residual = vsub(updated_positions[effector], target)
    return updated, residual, limit_conflicts


def _root_correction(
    skeleton: Skeleton,
    positions: dict[str, Vec],
    effector: str,
    anchor: Vec,
) -> Vec:
    """Move the root so the contact anchor stays within reach.

    The minimum translation along the base-to-anchor direction keeps the
    original body trajectory while bringing the anchor into reach.
    """
    base, mid = effector_chain(skeleton, effector)
    max_reach = vlength(skeleton.joints[mid].offset) + vlength(skeleton.joints[effector].offset) - 1e-6
    gap = vsub(anchor, positions[base])
    distance = vlength(gap)
    if distance <= max_reach:
        return (0.0, 0.0, 0.0)
    return vscale(gap, (distance - max_reach) / max(distance, 1e-12))


def _chain_reach(skeleton: Skeleton, effector: str) -> float:
    base, mid = effector_chain(skeleton, effector)
    return vlength(skeleton.joints[mid].offset) + vlength(skeleton.joints[effector].offset)


def _solve_frame(
    skeleton: Skeleton,
    frame: dict[str, Any],
    active: list[tuple[ContactInterval, float, Vec]],
) -> dict[str, Any]:
    """Resolve every active contact on one frame together.

    Root corrections are settled before any limb IK: higher priority contacts
    (smaller priority value) claim reach first, each correction is evaluated
    against the cumulative root move, and a contact that would pull a stronger
    one out of reach is explicitly conceded instead of silently dropped.
    IK then runs over all chains until converged, so solving a later limb
    never drags an already-pinned effector away.
    """
    order = sorted(active, key=lambda item: (item[0].priority, item[0].joint))
    initial_positions = pose_world_positions(skeleton, frame)
    free_positions = {interval.joint: initial_positions[interval.joint] for interval, _, _ in order}
    conceded: dict[str, list[str]] = {interval.joint: [] for interval, _, _ in order}
    root_shift = (0.0, 0.0, 0.0)
    claimed: list[tuple[str, Vec]] = []

    for interval, weight, full_target in order:
        positions = pose_world_positions(skeleton, frame)
        base, _ = effector_chain(skeleton, interval.joint)
        projected_base = vadd(positions[base], root_shift)
        gap = vsub(full_target, projected_base)
        reach = _chain_reach(skeleton, interval.joint) - 1e-6
        distance = vlength(gap)
        candidate = (0.0, 0.0, 0.0)
        if distance > reach:
            magnitude = distance - reach
            direction = vscale(gap, 1.0 / max(distance, 1e-12))
            candidate = vscale(direction, magnitude * weight)
        blocked_by: list[str] = []
        if vlength(candidate) > 1e-9 and weight >= 0.999:
            for stronger_joint, stronger_target in claimed:
                stronger_base, _ = effector_chain(skeleton, stronger_joint)
                moved_base = vadd(
                    pose_world_positions(skeleton, frame)[stronger_base],
                    vadd(root_shift, candidate),
                )
                if vlength(vsub(stronger_target, moved_base)) > _chain_reach(skeleton, stronger_joint) + 1e-4:
                    blocked_by.append(stronger_joint)
        if blocked_by:
            conceded[interval.joint] = blocked_by
            continue
        root_shift = tuple(root_shift[i] + candidate[i] for i in range(3))
        claimed.append((interval.joint, full_target))

    if vlength(root_shift) > 1e-12:
        root = frame["root_translation"]
        frame["root_translation"] = [root[i] + root_shift[i] for i in range(3)]

    blended_targets = {
        interval.joint: vlerp(free_positions[interval.joint], full_target, weight)
        for interval, weight, full_target in order
    }
    frame_errors: dict[str, dict[str, Any]] = {}
    for iteration in range(5):
        max_residual = 0.0
        for interval, weight, full_target in order:
            target = blended_targets[interval.joint]
            positions = pose_world_positions(skeleton, frame)
            globals_ = _ordered_frame_rotations(skeleton, frame)
            globals_, residual, limit_conflicts = _apply_two_bone_ik(
                skeleton, globals_, positions, interval.joint, target
            )
            base, mid = effector_chain(skeleton, interval.joint)
            locals_ = local_frame_rotations(skeleton, globals_)
            for name in (base, mid, interval.joint):
                frame["rotations"][name] = [float(value) for value in locals_[name]]
            error = vlength(residual)
            max_residual = max(max_residual, error)
            frame_errors[interval.joint] = {
                "error": error,
                "residual": list(residual),
                "target": list(target),
                "full_target": list(full_target),
                "limit_conflicts": limit_conflicts,
                "conceded_to": conceded[interval.joint],
                "weight": weight,
                "root_shift": list(root_shift),
            }
        if max_residual <= 1e-7:
            break

    for interval, weight, full_target in order:
        positions = pose_world_positions(skeleton, frame)
        residual = vsub(positions[interval.joint], blended_targets[interval.joint])
        frame_errors[interval.joint]["error"] = vlength(residual)
        frame_errors[interval.joint]["residual"] = list(residual)
    return frame_errors



def _chain_reach(skeleton: Skeleton, effector: str) -> float:
    base, mid = effector_chain(skeleton, effector)
    return vlength(skeleton.joints[mid].offset) + vlength(skeleton.joints[effector].offset)


def _solve_frame(
    skeleton: Skeleton,
    frame: dict[str, Any],
    active: list[tuple[ContactInterval, float, Vec]],
) -> dict[str, Any]:
    """Resolve every active contact on one frame together.

    Active contacts share the root translation; higher priority contacts
    (smaller priority value) claim reach first, and a contact that would pull
    a stronger one out of reach is explicitly conceded instead of silently
    dropped.  Root corrections are settled before limb IK so solving a later
    chain never drags an already-pinned effector away.
    """
    order = sorted(active, key=lambda item: (item[0].priority, item[0].joint))
    initial_positions = pose_world_positions(skeleton, frame)
    free_positions = {interval.joint: initial_positions[interval.joint] for interval, _, _ in order}
    conceded: dict[str, list[str]] = {interval.joint: [] for interval, _, _ in order}
    root_shift = (0.0, 0.0, 0.0)

    claimed: list[tuple[ContactInterval, Vec, float]] = []
    for interval, weight, full_target in order:
        positions = pose_world_positions(skeleton, frame)
        needed = _root_correction(skeleton, positions, interval.joint, full_target)
        desired_correction = vscale(needed, weight)
        blocked_by: list[str] = []
        if vlength(desired_correction) > 1e-9:
            for stronger, stronger_target, stronger_weight in claimed:
                if stronger_weight < 0.999:
                    continue
                positions = pose_world_positions(skeleton, frame)
                stronger_base, _ = effector_chain(skeleton, stronger.joint)
                moved_base = vadd(positions[stronger_base], desired_correction)
                if vlength(vsub(stronger_target, moved_base)) > _chain_reach(skeleton, stronger.joint) + 1e-4:
                    blocked_by.append(stronger.joint)
        if blocked_by:
            conceded[interval.joint] = blocked_by
            continue
        root = frame["root_translation"]
        frame["root_translation"] = [root[i] + desired_correction[i] for i in range(3)]
        root_shift = tuple(root_shift[i] + desired_correction[i] for i in range(3))
        claimed.append((interval, full_target, weight))

    shifted_free = dict(free_positions)
    blended_targets = {
        interval.joint: vlerp(shifted_free[interval.joint], full_target, weight)
        for interval, weight, full_target in order
    }
    frame_errors: dict[str, dict[str, Any]] = {}
    for iteration in range(4):
        max_residual = 0.0
        for interval, weight, full_target in order:
            target = blended_targets[interval.joint]
            positions = pose_world_positions(skeleton, frame)
            globals_ = _ordered_frame_rotations(skeleton, frame)
            globals_, residual, limit_conflicts = _apply_two_bone_ik(
                skeleton, globals_, positions, interval.joint, target
            )
            base, mid = effector_chain(skeleton, interval.joint)
            locals_ = local_frame_rotations(skeleton, globals_)
            for name in (base, mid, interval.joint):
                frame["rotations"][name] = [float(value) for value in locals_[name]]
            error = vlength(residual)
            max_residual = max(max_residual, error)
            frame_errors[interval.joint] = {
                "error": error,
                "residual": list(residual),
                "target": list(target),
                "full_target": list(full_target),
                "limit_conflicts": limit_conflicts,
                "conceded_to": conceded[interval.joint],
                "weight": weight,
                "root_shift": list(root_shift),
            }
        if max_residual <= 1e-7:
            break

    for interval, weight, full_target in order:
        positions = pose_world_positions(skeleton, frame)
        residual = vsub(positions[interval.joint], blended_targets[interval.joint])
        frame_errors[interval.joint]["error"] = vlength(residual)
        frame_errors[interval.joint]["residual"] = list(residual)
    return frame_errors


def apply_contact_constraints(
    target: Skeleton,
    motion_data: dict[str, Any],
    *,
    ground_scale: float = 1.0,
    contact_plan: list[dict[str, Any]] | None = None,
    plan_name: str = "default",
) -> dict[str, Any]:
    """Satisfy all declared contacts, including overlapping multi-limb ones.

    Contacts may target any two-bone end effector (feet and hands).  Moving
    supports are described with per-frame target keyframes, entry/exit blend
    windows ease the effector onto and off its target, and frames outside all
    contact intervals keep playing the retargeted motion untouched.  When
    constraints cannot all be met, per-contact errors and the higher-priority
    contacts each one yielded to are reported instead of faking success.
    """
    frames = motion_data["frames"]
    effectors = {
        joint.name
        for joint in target.joints.values()
        if joint.parent is not None
        and target.joints[joint.parent].parent is not None
        and joint.name.endswith(("_foot", "_hand"))
    }
    raw_contacts = list(motion_data.get("contacts", [])) if contact_plan is None else contact_plan
    intervals = parse_contacts(raw_contacts, len(frames), effectors)
    if not intervals:
        motion_data["contacts"] = []
        motion_data["grounding"] = {
            "plan": plan_name,
            "policy": "no contacts declared; retargeted motion plays untouched",
            "intervals": [],
        }
        return {
            "plan": plan_name,
            "contacts": [],
            "unmet": [],
            "concessions": [],
            "max_contact_error": 0.0,
        }

    anchors: dict[int, Vec] = {}
    for interval in intervals:
        start_world = pose_world_positions(target, frames[interval.start])[interval.joint]
        if interval.keyframes:
            first_key = min(interval.keyframes, key=lambda item: item.frame)
            anchor = tuple(value * ground_scale for value in first_key.position)
        else:
            anchor = (start_world[0], interval.ground_y * ground_scale, start_world[2])
        anchors[interval.index] = anchor

    unmet: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    max_error = 0.0
    concessions: list[dict[str, Any]] = []

    intervals_by_frame: dict[int, list[tuple[ContactInterval, float, Vec]]] = {}
    for frame_index in range(len(frames)):
        active = []
        for interval in intervals:
            weight = interval.weight(frame_index)
            if weight <= 0.0:
                continue
            anchor = anchors[interval.index]
            full_target = (
                interval.target_position(frame_index, None, ground_scale)
                if interval.keyframes
                else anchor
            )
            active.append((interval, weight, full_target))
        if active:
            intervals_by_frame[frame_index] = active

    for frame_index in sorted(intervals_by_frame):
        frame_errors = _solve_frame(target, frames[frame_index], intervals_by_frame[frame_index])
        for interval, weight, full_target in intervals_by_frame[frame_index]:
            outcome = frame_errors[interval.joint]
            error = outcome["error"]
            if weight >= 1.0:
                max_error = max(max_error, error)
            if error > CONTACT_TOLERANCE or outcome["limit_conflicts"]:
                if weight < 1.0 and not outcome["limit_conflicts"]:
                    continue
                reason = "joint limit conflict" if outcome["limit_conflicts"] else "outside limb reach"
                if outcome["conceded_to"]:
                    reason = "conceded to higher-priority contact"
                    concessions.append({
                        "type": "contact_concession",
                        "frame": frame_index,
                        "joint": interval.joint,
                        "conceded_to": outcome["conceded_to"],
                        "error": error,
                    })
                unmet.append({
                    "type": "unmet_contact",
                    "frame": frame_index,
                    "joint": interval.joint,
                    "priority": interval.priority,
                    "weight": weight,
                    "anchor": list(full_target),
                    "residual": outcome["residual"],
                    "error": error,
                    "root_adjustment": outcome["root_shift"],
                    "limit_conflicts": outcome["limit_conflicts"],
                    "conceded_to": outcome["conceded_to"],
                    "reason": reason,
                })

    for interval in intervals:
        interval_unmet = [item for item in unmet if item["frame"] in interval.frames() and item["joint"] == interval.joint]
        interval_max = max((item["error"] for item in interval_unmet), default=0.0)
        anchor = anchors[interval.index]
        status = "satisfied"
        if any(item["conceded_to"] for item in interval_unmet):
            status = "conceded"
        elif interval_unmet:
            status = "unmet"
        reports.append({
            "joint": interval.joint,
            "start": interval.start,
            "end": interval.end,
            "ground_y": interval.ground_y * ground_scale,
            "priority": interval.priority,
            "blend_in": interval.blend_in,
            "blend_out": interval.blend_out,
            "moving": bool(interval.keyframes),
            "anchor": list(anchor),
            "world_keyframes": [
                {"frame": key.frame, "position": [value * ground_scale for value in key.position]}
                for key in sorted(interval.keyframes, key=lambda item: item.frame)
            ],
            "max_error": interval_max,
            "unmet_frames": [item["frame"] for item in interval_unmet],
            "conceded_to": sorted(
                {joint for item in interval_unmet for joint in item["conceded_to"]}
            ),
            "status": status,
        })

    motion_data["contacts"] = [
        {
            "joint": report["joint"],
            "start": report["start"],
            "end": report["end"],
            "ground_y": report["ground_y"],
            "priority": report["priority"],
            "blend_in": report["blend_in"],
            "blend_out": report["blend_out"],
            "moving": report["moving"],
            "anchor": report["anchor"],
            "target_keyframes": report["world_keyframes"],
        }
        for report in reports
    ]
    motion_data["grounding"] = {
        "plan": plan_name,
        "policy": "overlapping end-effector contacts resolved with shared root correction; "
        "higher-priority targets claimed first; entry/exit blends ease contact; "
        "moving targets follow keyframes; concessions and residuals are reported",
        "intervals": reports,
        "max_contact_error": max_error,
    }
    return {
        "plan": plan_name,
        "contacts": reports,
        "unmet": unmet,
        "concessions": concessions,
        "max_contact_error": max_error,
    }


def apply_foot_contacts(
    target: Skeleton,
    motion_data: dict[str, Any],
    *,
    ground_scale: float = 1.0,
) -> dict[str, Any]:
    """Backward-compatible entry point for the legacy single-foot grounding API."""
    return apply_contact_constraints(target, motion_data, ground_scale=ground_scale)
