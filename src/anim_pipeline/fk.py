from __future__ import annotations

import numpy as np

from .models import Animation, Skeleton
from .math3d import multiply, rotate


def forward_kinematics(
    skeleton: Skeleton, local_rotations: dict[str, np.ndarray], root_translation: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    positions: dict[str, np.ndarray] = {}
    world_rotations: dict[str, np.ndarray] = {}
    for name in skeleton.children_order():
        joint = skeleton.joints[name]
        local_rotation = local_rotations.get(name, joint.rotation)
        if joint.parent is None:
            world_rotations[name] = local_rotation.copy()
            positions[name] = np.asarray(root_translation, dtype=float) + rotate(
                local_rotation, joint.translation
            )
        else:
            parent_rotation = world_rotations[joint.parent]
            world_rotations[name] = multiply(parent_rotation, local_rotation)
            positions[name] = positions[joint.parent] + rotate(parent_rotation, joint.translation)
    return positions, world_rotations


def animation_world_pose(
    skeleton: Skeleton, animation: Animation, frame: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    return forward_kinematics(
        skeleton, animation.local_rotations[frame], animation.root_translations[frame]
    )


def rest_pose(skeleton: Skeleton) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rotations = {name: joint.rotation.copy() for name, joint in skeleton.joints.items()}
    return forward_kinematics(skeleton, rotations, np.zeros(3))


def chain_length(skeleton: Skeleton, start: str, end: str, positions=None) -> float:
    chain = [end]
    current = end
    while current != start:
        parent = skeleton.joints[current].parent
        if parent is None:
            raise ValueError(f"{end!r} is not a descendant of {start!r}")
        current = parent
        chain.append(current)
    chain.reverse()
    return float(sum(np.linalg.norm(skeleton.joints[child].translation) for child in chain[1:]))
