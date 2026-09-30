from __future__ import annotations

from copy import deepcopy
from math import acos, cos, sin
from typing import Any

from .math3d import (
    Vec,
    clamp,
    euler_xyz_to_quat,
    q_rotate_vector,
    qidentity,
    qinverse,
    qmul,
    quat_shortest_arc,
    quat_to_euler_xyz,
    vadd,
    vcross,
    vdist,
    vdot,
    vlength,
    vnorm,
    vscale,
    vsub,
)
from .models import Motion, Skeleton
from .retarget import global_frame_rotations, local_frame_rotations, pose_world_positions


def _chain(skeleton: Skeleton, foot: str) -> list[str]:
    result: list[str] = []
    current: str | None = foot
    while current is not None:
        result.append(current)
        current = skeleton.joints[current].parent
    result.reverse()
    return result


def _fk(skeleton: Skeleton, root: Vec, globals_: dict[str, Any]) -> dict[str, Vec]:
    positions: dict[str, Vec] = {}
    for name in skeleton.ordered():
        joint = skeleton.joints[name]
        if joint.parent is None:
            positions[name] = tuple(root)
        else:
            positions[name] = tuple(
                positions[joint.parent][i] + q_rotate_vector(globals_[joint.parent], joint.offset)[i]
                for i in range(3)
            )
    return positions


def _frame_relative_euler(skeleton: Skeleton, joint: str, parent_global: Any, global_rotation: Any):
    bind = skeleton.joints[joint].bind_rotation or qidentity()
    frame_global = qmul(parent_global, bind)
    return quat_to_euler_xyz(qmul(qinverse(frame_global), global_rotation))


def _apply_frame_relative_euler(skeleton: Skeleton, joint: str, parent_global: Any, euler):
    bind = skeleton.joints[joint].bind_rotation or qidentity()
    frame_global = qmul(parent_global, bind)
    values = list(euler)
    limits = skeleton.joints[joint].rotation_limits
    if limits:
        for axis, index in (("x", 0), ("y", 1), ("z", 2)):
            if axis in limits:
                low, high = limits[axis]
                values[index] = clamp(values[index], low, high)
    return qmul(frame_global, euler_xyz_to_quat(tuple(values)))


def _solve_leg(skeleton, globals_, root, chain, target):
    hip, knee, foot = chain[-3], chain[-2], chain[-1]
    upper = vlength(skeleton.joints[knee].offset)
    lower = vlength(skeleton.joints[foot].offset)
    positions = _fk(skeleton, root, globals_)
    hip_point = positions[hip]
    to_target = vsub(target, hip_point)
    distance = vlength(to_target)
    if distance < 1e-9:
        return globals_, positions, distance
    reachable = min(max(distance, abs(upper - lower) + 1e-6), upper + lower - 1e-6)
    aim = vscale(vnorm(to_target), reachable)
    original_upper = vsub(positions[knee], hip_point)
    projected = vsub(original_upper, vscale(aim, vdot(original_upper, aim) / max(vdot(aim, aim), 1e-12)))
    if vlength(projected) < 1e-9:
        projected = vcross(aim, (0.0, 1.0, 0.0))
    normal = vnorm(vcross(vnorm(aim), vnorm(projected)))
    cos_hip = clamp((upper * upper + reachable * reachable - lower * lower) / (2 * upper * reachable), -1, 1)
    cos_knee = clamp((upper * upper + lower * lower - reachable * reachable) / (2 * upper * lower), -1, 1)
    hip_angle = acos(cos_hip)
    desired_upper = vadd(
        vscale(vnorm(aim), upper * cos(hip_angle)),
        vscale(normal, upper * sin(hip_angle)),
    )
    parent_name = skeleton.joints[hip].parent
    hip_parent = globals_[parent_name] if parent_name else qidentity()
    hip_world = quat_shortest_arc(vnorm(original_upper), vnorm(desired_upper))
    hip_euler = _frame_relative_euler(skeleton, hip, hip_parent, qmul(hip_world, globals_[hip]))
    globals_[hip] = _apply_frame_relative_euler(skeleton, hip, hip_parent, hip_euler)
    positions = _fk(skeleton, root, globals_)
    shin = vsub(positions[foot], positions[knee])
    wanted = vsub(target, positions[knee])
    knee_world = quat_shortest_arc(vnorm(shin), vnorm(wanted))
    knee_euler = _frame_relative_euler(skeleton, knee, globals_[hip], qmul(knee_world, globals_[knee]))
    globals_[knee] = _apply_frame_relative_euler(skeleton, knee, globals_[hip], knee_euler)
    positions = _fk(skeleton, root, globals_)
    return globals_, positions, vdist(positions[foot], target)


def solve_contacts(skeleton, motion, source_motion=None, foot_pairs=None, tolerance=0.01):
    frames = deepcopy(motion.frames)
    if foot_pairs is None:
        pairs = {name: name for name in motion.contacts}
    else:
        assert source_motion is not None
        pairs = {source: target for source, target in foot_pairs.items() if source in source_motion.contacts}
    chains = {target: _chain(skeleton, target) for target in pairs.values()}
    interval_records = {}
    active = {}
    for source_foot, target_foot in pairs.items():
        intervals = source_motion.contacts[source_foot] if source_motion is not None else motion.contacts[source_foot]
        records = []
        for interval in intervals:
            start, end = int(interval["start"]), int(interval["end"])
            anchor = pose_world_positions(skeleton, frames[start])[target_foot]
            records.append((start, end, anchor))
            for frame_index in range(start, end + 1):
                active.setdefault(frame_index, []).append((target_foot, anchor))
        interval_records[target_foot] = records

    unmet = []
    for frame_index, contacts in active.items():
        frame = frames[frame_index]
        globals_ = global_frame_rotations(skeleton, frame)
        root_translation = tuple(frame["root_translation"])
        changed = set()
        residuals = {}
        for _round in range(3):
            positions = _fk(skeleton, root_translation, globals_)
            shifts = [vsub(anchor, positions[foot]) for foot, anchor in contacts]
            root_translation = vadd(
                root_translation,
                tuple(sum(v[i] for v in shifts) / len(shifts) for i in range(3)),
            )
            changed = set()
            residuals = {}
            for target_foot, anchor in contacts:
                globals_, positions, residual = _solve_leg(
                    skeleton, globals_, root_translation, chains[target_foot], anchor
                )
                residuals[target_foot] = (positions[target_foot], residual)
                changed.update(chains[target_foot][-3:-1])
        locals_ = local_frame_rotations(skeleton, globals_)
        frame["root_translation"] = [float(v) for v in root_translation]
        rotations = dict(frame["rotations"])
        for joint in changed:
            rotations[joint] = [float(v) for v in locals_[joint]]
        frame["rotations"] = rotations
        for target_foot, anchor in contacts:
            actual, residual = residuals[target_foot]
            if residual > tolerance:
                unmet.append({"type": "unmet_contact", "frame": frame_index, "joint": target_foot,
                              "anchor": list(anchor), "actual": list(actual), "residual": residual,
                              "reason": "joint limits or leg reach prevent full grounding; pose preserved"})

    slides = []
    for target_foot, records in interval_records.items():
        for start, end, anchor in records:
            start_point = pose_world_positions(skeleton, frames[start])[target_foot]
            end_point = pose_world_positions(skeleton, frames[end])[target_foot]
            residual_map = {
                item["frame"]: item["residual"]
                for item in unmet if item["joint"] == target_foot
            }
            max_slide = max(
                (vdist(pose_world_positions(skeleton, frames[i])[target_foot], anchor)
                 for i in range(start, end + 1)),
                default=0.0,
            )
            slides.append({"joint": target_foot, "start": start, "end": end,
                           "anchor": list(anchor), "slide": max_slide,
                           "reachable_residual": max(
                               (residual_map.get(i, 0.0) for i in range(start, end + 1)),
                               default=0.0,
                           )})
    return {"frames": frames, "anchors": {
        foot: [{"start": s, "end": e, "anchor": list(a)} for s, e, a in records]
        for foot, records in interval_records.items()
    }, "unmet": unmet, "slides": slides}
