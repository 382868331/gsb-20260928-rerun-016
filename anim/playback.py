from __future__ import annotations

from typing import Any

from .models import Motion, Skeleton
from .retarget import pose_world_positions


def play_pose_samples(skeleton: Skeleton, motion: Motion) -> list[dict[str, Any]]:
    frames = []
    for index, frame in enumerate(motion.frames):
        frames.append({
            "index": index,
            "time": index / motion.frame_rate,
            "root_translation": frame["root_translation"],
            "joint_world_positions": pose_world_positions(skeleton, frame),
        })
    return frames
