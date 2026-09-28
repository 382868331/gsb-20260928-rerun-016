from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .math3d import Vec, qidentity, qinverse, qmul, q_rotate_vector, vlength, vsub
from .models import Mapping, Motion, Skeleton


@dataclass
class RetargetResult:
    document: dict[str, Any]
    ignored_source: list[str]
    static_target: list[str]


def skeleton_height(skeleton: Skeleton) -> float:
    return sum(vlength(joint.offset) for joint in skeleton.joints.values())


def mapped_skeleton_scale(source: Skeleton, target: Skeleton, pairs: dict[str, str]) -> float:
    source_length = sum(vlength(source.joints[name].offset) for name in pairs)
    target_length = sum(vlength(target.joints[name].offset) for name in pairs.values())
    return target_length / source_length


def global_bind_rotations(skeleton: Skeleton) -> dict[str, Any]:
    result = {}
    for name in skeleton.ordered():
        joint = skeleton.joints[name]
        local = joint.bind_rotation if joint.bind_rotation is not None else qidentity()
        if joint.parent is None:
            result[name] = local
        else:
            result[name] = qmul(result[joint.parent], local)
    return result


def global_frame_rotations(
    skeleton: Skeleton,
    frame: dict[str, Any],
    bind: dict[str, Any],
) -> dict[str, Any]:
    result = {}
    rotations = frame["rotations"]
    for name in skeleton.ordered():
        joint = skeleton.joints[name]
        local = tuple(rotations.get(name, (1, 0, 0, 0)))
        bind = joint.bind_rotation if joint.bind_rotation is not None else qidentity()
        relative = qmul(bind, local)
        if joint.parent is None:
            result[name] = relative
        else:
            result[name] = qmul(result[joint.parent], relative)
    return result


def local_frame_rotations(
    skeleton: Skeleton,
    globals_: dict[str, Any],
) -> dict[str, Any]:
    result = {}
    for name in skeleton.ordered():
        joint = skeleton.joints[name]
        global_delta = globals_[name]
        if joint.parent is None:
            parent_global = qidentity()
        else:
            parent_global = globals_[joint.parent]
        bind = joint.bind_rotation if joint.bind_rotation is not None else qidentity()
        result[name] = qmul(qinverse(bind), qmul(qinverse(parent_global), global_delta))
    return result


def retarget_motion(source: Skeleton, target: Skeleton, motion: Motion, mapping: Mapping) -> RetargetResult:
    if mapping.source_skeleton != source.name or mapping.target_skeleton != target.name:
        raise ValueError("mapping does not match supplied skeletons")
    if motion.skeleton != source.name:
        raise ValueError("motion does not belong to source skeleton")

    pairs = mapping.pairs()
    required = mapping.required_pairs()
    required_targets = set(required.values())
    mapped_sources = set(pairs)
    missing_animated: list[str] = []
    for frame in motion.frames:
        for name in frame["rotations"]:
            if name not in mapped_sources:
                missing_animated.append(name)
    if missing_animated:
        missing = sorted(set(missing_animated))
        raise ValueError(f"animated joints have no mapping: {', '.join(missing)}")

    source_bind = global_bind_rotations(source)
    target_bind = global_bind_rotations(target)
    scale = mapped_skeleton_scale(source, target, required)
    target_global_frames: list[dict[str, Any]] = []

    for frame_index, frame in enumerate(motion.frames):
        source_global = global_frame_rotations(source, frame, source_bind)
        target_global = {name: rotation for name, rotation in target_bind.items()}
        for source_name, target_name in required.items():
            if source_name not in source_global:
                raise ValueError(f"mapping references missing source joint {source_name}")
            if target_name not in target_bind:
                raise ValueError(f"mapping references missing target joint {target_name}")
            source_delta = qmul(source_global[source_name], qinverse(source_bind[source_name]))
            target_global[target_name] = qmul(source_delta, target_bind[target_name])
        target_global_frames.append(target_global)

    output_frames = []
    for frame_index, (source_frame, target_global) in enumerate(zip(motion.frames, target_global_frames)):
        target_local = local_frame_rotations(target, target_global)
        source_translation = source_frame["root_translation"]
        output_frames.append({
            "root_translation": [source_translation[0] * scale, source_translation[1] * scale, source_translation[2] * scale],
            "rotations": {
                name: [float(component) for component in rotation]
                for name, rotation in target_local.items()
                if name in required_targets or any(
                    component != (1.0 if index == 0 else 0.0)
                    for index, component in enumerate(rotation)
                )
            },
        })

    ignored_source = sorted(name for name, target_name in pairs.items() if target_name is None)
    static_target = sorted(name for name in target.joints if name not in required_targets)
    document = {
        "name": f"{motion.name}_retargeted_to_{target.name}",
        "kind": "retargeted_motion",
        "source_motion": motion.name,
        "source_skeleton": source.name,
        "target_skeleton": target.name,
        "skeleton": target.name,
        "mapping": mapping.name,
        "frame_rate": motion.frame_rate,
        "root_scale": scale,
        "frames": output_frames,
        "retargeting": {
            "method": "bind-pose-relative global rotation transfer",
            "missing_source_joints": ignored_source,
            "target_only_joints": static_target,
            "bone_length_policy": "rotations transferred; root translation scaled by mapped joint bone length",
        },
    }
    return RetargetResult(document, ignored_source, static_target)


def pose_world_positions(skeleton: Skeleton, frame: dict[str, Any]) -> dict[str, Vec]:
    bind = global_bind_rotations(skeleton)
    rotations = global_frame_rotations(skeleton, frame, bind)
    positions = {}
    for name in skeleton.ordered():
        joint = skeleton.joints[name]
        if joint.parent is None:
            positions[name] = tuple(frame["root_translation"])
        else:
            parent_position = positions[joint.parent]
            rotated_offset = q_rotate_vector(rotations[joint.parent], joint.offset)
            positions[name] = tuple(parent_position[i] + rotated_offset[i] for i in range(3))
    return positions


def source_to_target_position(source_position: Vec, source_origin: Vec, target_origin: Vec, scale: float) -> Vec:
    delta = vsub(source_position, source_origin)
    return tuple(target_origin[i] + delta[i] * scale for i in range(3))
