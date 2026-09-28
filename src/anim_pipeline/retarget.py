from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .fk import chain_length, forward_kinematics, rest_pose
from .models import Animation, Mapping, Skeleton
from .math3d import axis_angle, inverse, multiply, rotate, rotation_between


@dataclass
class RetargetResult:
    animation: Animation
    unmapped_source: list[str]
    unmapped_target: list[str]
    scale: float


def _leg_scale(source: Skeleton, target: Skeleton) -> float:
    def leg_length(skeleton: Skeleton) -> float | None:
        values = []
        for prefix in ("left", "right"):
            upper = f"{prefix}_upleg"
            lower = f"{prefix}_leg"
            foot = f"{prefix}_foot"
            if upper in skeleton.joints and lower in skeleton.joints and foot in skeleton.joints:
                values.append(chain_length(skeleton, upper, lower) + chain_length(skeleton, lower, foot))
        return float(np.mean(values)) if values else None

    source_length = leg_length(source)
    target_length = leg_length(target)
    if source_length is None or target_length is None:
        return 1.0
    return float(target_length / source_length)


def retarget(
    source: Skeleton,
    target: Skeleton,
    animation: Animation,
    mapping: Mapping,
) -> RetargetResult:
    if source.name != mapping.source_skeleton or target.name != mapping.target_skeleton:
        raise ValueError("mapping skeleton names do not match source and target")
    if animation.skeleton_name != source.name:
        raise ValueError("animation does not belong to source skeleton")

    invalid_source = sorted(set(mapping.pairs) - set(source.joints))
    invalid_target = sorted(
        target_name
        for target_name in mapping.pairs.values()
        if target_name is not None and target_name not in target.joints
    )
    unmapped_targets = sorted(n for n in mapping.required_targets if mapping.inverse_pairs.get(n) is None)
    if invalid_source or invalid_target or unmapped_targets:
        raise ValueError(
            f"incomplete joint mapping: missing_source={invalid_source}, "
            f"missing_target={invalid_target}, unmapped_required={unmapped_targets}"
        )

    scale = _leg_scale(source, target) if mapping.scale_mode == "leg_length" else 1.0
    source_rest_positions, source_rest_world = rest_pose(source)
    target_rest_positions, target_rest_world = rest_pose(target)

    source_root_rest = source_rest_positions[source.root_name].copy()
    target_root_rest = target_rest_positions[target.root_name].copy()
    foot_pairs: list[tuple[str, str]] = []
    for foot_name in ("left_foot", "right_foot"):
        target_foot = mapping.pairs.get(foot_name)
        if foot_name in source.joints and target_foot in target.joints:
            foot_pairs.append((foot_name, target_foot))

    target_frames = []
    for frame_index in range(animation.frame_count):
        source_positions, source_world = forward_kinematics(
            source,
            animation.local_rotations[frame_index],
            animation.root_translations[frame_index],
        )
        mapped_world: dict[str, np.ndarray] = {}
        target_local: dict[str, np.ndarray] = {}
        for source_name, target_name in mapping.pairs.items():
            if target_name is None:
                continue
            delta = multiply(
                source_world[source_name], inverse(source_rest_world[source_name])
            )
            mapped_world[target_name] = multiply(
                delta, target_rest_world[target_name]
            )
        source_delta = animation.root_translations[frame_index].copy() * scale
        root_translation = target_root_rest + source_delta
        target_chain_world = {
            target_name: world_rotation
            for target_name, world_rotation in mapped_world.items()
        }
        for source_foot, target_foot in foot_pairs:
            target_knee = target.joints[target_foot].parent
            target_hip = target.joints[target_knee].parent
            if target_knee is None or target_hip is None:
                continue
            if not all(name in mapped_world for name in (target_hip, target_knee, target_foot)):
                continue
            desired_foot = target_root_rest + scale * (
                source_positions[source_foot] - source_root_rest
            )
            chain_positions: dict[str, np.ndarray] = {}
            chain_rotations: dict[str, np.ndarray] = {}
            for name in target.children_order():
                joint = target.joints[name]
                world_rotation = target_chain_world.get(name, target_rest_world[name])
                chain_rotations[name] = world_rotation
                if joint.parent is None:
                    chain_positions[name] = root_translation + rotate(world_rotation, joint.translation)
                else:
                    chain_positions[name] = chain_positions[joint.parent] + rotate(
                        chain_rotations[joint.parent], joint.translation
                    )
            hip_position = chain_positions[target_hip]
            upper = float(np.linalg.norm(target_rest_positions[target_knee] - target_rest_positions[target_hip]))
            lower = float(np.linalg.norm(target_rest_positions[target_foot] - target_rest_positions[target_knee]))
            to_foot = desired_foot - hip_position
            distance = float(np.linalg.norm(to_foot))
            reach = upper + lower - 1e-6
            distance = min(max(distance, abs(upper - lower) + 1e-6), reach)
            foot_dir = to_foot / max(float(np.linalg.norm(to_foot)), 1e-9)
            pole_axis = np.cross(foot_dir, np.array([1.0, 0.0, 0.0]))
            if np.linalg.norm(pole_axis) < 1e-9:
                pole_axis = np.cross(foot_dir, np.array([0.0, 1.0, 0.0]))
            pole_axis = pole_axis / np.linalg.norm(pole_axis)
            hinge_angle = np.arccos(np.clip((upper * upper + distance * distance - lower * lower) / (2 * upper * distance), -1, 1))
            knee_position = hip_position + upper * rotate(axis_angle(pole_axis, hinge_angle), foot_dir)
            hip_world = rotation_between(
                target_rest_positions[target_knee] - target_rest_positions[target_hip],
                knee_position - hip_position,
            )
            knee_world = rotation_between(
                target_rest_positions[target_foot] - target_rest_positions[target_knee],
                desired_foot - knee_position,
            )
            mapped_world[target_hip] = hip_world
            mapped_world[target_knee] = knee_world
            target_chain_world[target_hip] = hip_world
            target_chain_world[target_knee] = knee_world
            chain_positions[target_knee] = knee_position
            chain_positions[target_foot] = desired_foot

        for target_name in target.children_order():
            if target_name in mapped_world:
                world_rotation = mapped_world[target_name]
            else:
                world_rotation = target_rest_world[target_name].copy()
            parent = target.joints[target_name].parent
            parent_world = mapped_world[parent] if parent is not None and parent in mapped_world else (
                target_rest_world[parent] if parent is not None else None
            )
            if target_name == target.root_name:
                target_local[target_name] = world_rotation.copy()
            elif parent_world is None:
                target_local[target_name] = target.joints[target_name].rotation.copy()
            else:
                target_local[target_name] = multiply(inverse(parent_world), world_rotation)

        target_frames.append((root_translation, target_local))

    converted = Animation(
        name=f"{animation.name}_retargeted_to_{target.name}",
        skeleton_name=target.name,
        frame_time=animation.frame_time,
        root_translations=[item[0] for item in target_frames],
        local_rotations=[item[1] for item in target_frames],
        metadata={
            "source_animation": animation.name,
            "source_skeleton": source.name,
            "target_skeleton": target.name,
            "mapping": mapping.name,
            "root_translation_scale": scale,
            "original_record": {
                "skeleton": source.name,
                "frame_count": animation.frame_count,
                "frame_time": animation.frame_time,
            },
        },
    )
    mapped_source = {source_name for source_name, target_name in mapping.pairs.items() if target_name is not None}
    unmapped_source = sorted(set(source.joints) - mapped_source)
    mapped_targets = {target for target in mapping.pairs.values() if target is not None}
    unmapped_target = sorted(set(target.joints) - mapped_targets)
    return RetargetResult(converted, unmapped_source, unmapped_target, scale)
