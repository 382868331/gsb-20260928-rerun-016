from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .math3d import Vec, qidentity, qinverse, qmul, q_rotate_vector, vlength, vsub
from .math3d import Quat
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


def bind_frame_rotation(source: Skeleton, target: Skeleton) -> Quat:
    """Rotation taking source bind axes to target bind axes at the root."""
    source_root = source.joints[source.root].bind_rotation or qidentity()
    target_root = target.joints[target.root].bind_rotation or qidentity()
    return qmul(target_root, qinverse(source_root))


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
    frame_change = bind_frame_rotation(source, target)
    inverse_frame_change = qinverse(frame_change)
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
            target_delta = qmul(qmul(frame_change, source_delta), inverse_frame_change)
            target_global[target_name] = qmul(target_delta, target_bind[target_name])
        target_global_frames.append(target_global)

    output_frames = []
    for frame_index, (source_frame, target_global) in enumerate(zip(motion.frames, target_global_frames)):
        target_local = local_frame_rotations(target, target_global)
        source_translation = source_frame["root_translation"]
        relative_root = (
            source_translation[0] - motion.frames[0]["root_translation"][0],
            source_translation[1] - motion.frames[0]["root_translation"][1],
            source_translation[2] - motion.frames[0]["root_translation"][2],
        )
        scaled_delta = tuple(component * scale for component in q_rotate_vector(frame_change, relative_root))
        target_first = tuple(component * scale for component in motion.frames[0]["root_translation"])
        output_frames.append({
            "root_translation": [target_first[i] + scaled_delta[i] for i in range(3)],
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

    target_contacts = []
    for contact in motion.contacts:
        target_foot = pairs.get(contact["joint"])
        if target_foot is None:
            continue
        converted = {
            "joint": target_foot,
            "start": int(contact["start"]),
            "end": int(contact["end"]),
            "ground_y": float(contact.get("ground_y", 0.0)),
        }
        for option in ("priority", "blend_in", "blend_out"):
            if option in contact:
                converted[option] = int(contact[option])
        if contact.get("target_keyframes"):
            converted["target_keyframes"] = [
                {
                    "frame": int(key["frame"]),
                    "position": [float(value) for value in key["position"]],
                }
                for key in contact["target_keyframes"]
            ]
        target_contacts.append(converted)

    document = {
        "name": f"{motion.name}_retargeted_to_{target.name}",
        "kind": "retargeted_motion",
        "source_motion": motion.name,
        "source_skeleton": source.name,
        "target_skeleton": target.name,
        "skeleton": target.name,
        "mapping": mapping.name,
        "mapping_version": mapping.version,
        "frame_rate": motion.frame_rate,
        "root_scale": scale,
        "contacts": target_contacts,
        "local_revisions": list(motion.local_revisions),
        "frames": output_frames,
        "retargeting": {
            "method": "bind-pose-relative global rotation transfer",
            "source_resource": motion.name,
            "missing_source_joints": ignored_source,
            "target_only_joints": static_target,
            "bone_length_policy": "rotations transferred; root translation scaled by mapped joint bone length",
            "local_revisions": list(motion.local_revisions),
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
