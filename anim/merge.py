from __future__ import annotations

from copy import deepcopy
from typing import Any

from .contact import solve_contacts
from .math3d import q_rotate_vector, quat_yaw, vdist, vsub
from .models import Motion, Skeleton
from .retarget import global_bind_rotations, global_frame_rotations, retarget_motion


def _world_yaw(rotation: Any) -> float:
    return quat_yaw(rotation)


def _apply_revisions(frame: dict[str, Any], revisions: list[dict[str, Any]], frame_index: int) -> None:
    for revision in revisions:
        at = int(revision.get("frame", frame_index))
        if at != frame_index:
            continue
        if "root_translation_delta" in revision:
            frame["root_translation"] = [
                frame["root_translation"][i] + float(revision["root_translation_delta"][i])
                for i in range(3)
            ]
        if "rotations" in revision:
            frame["rotations"].update(deepcopy(revision["rotations"]))



def merge_clips(
    clips: list[dict[str, Any]],
    *,
    name: str,
    target: Skeleton,
    ground_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Retarget and concatenate clips using each clip's own mapping.

    Each clip dict provides source/target skeletons, source motion, mapping and
    optional local ``revisions``. Continuous translation and turning are carried
    between clips by aligning each clip's first root frame to the prior clip's
    last frame. Ground anchors are transformed with the clip, so contact checks
    remain valid after concatenation.
    """
    output_frames: list[dict[str, Any]] = []
    output_contacts: dict[str, list[dict[str, int | str]]] = {}
    provenance: list[dict[str, Any]] = []
    cumulative_offset: list[float] = [0.0, 0.0, 0.0]
    cumulative_yaw = 0.0
    frame_offset = 0
    contact_reports: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    frame_rate: float | None = None
    mapping_versions: set[str] = set()
    source_resources: list[str] = []

    for clip_index, clip in enumerate(clips):
        source: Skeleton = clip["source_skeleton"]
        source_motion: Motion = clip["source_motion"]
        mapping = clip["mapping"]
        frame_rate = source_motion.frame_rate if frame_rate is None else frame_rate
        if abs(source_motion.frame_rate - frame_rate) > 1e-9:
            raise ValueError("cannot merge clips with different frame rates")
        mapping_versions.add(f"{mapping.name}@{mapping.version}")
        source_resources.append(clip.get("source_path", source_motion.name))

        result = retarget_motion(source, target, source_motion, mapping)
        document = result.document
        for local_index, frame in enumerate(document["frames"]):
            _apply_revisions(frame, clip.get("revisions", []), local_index)
        target_motion = Motion(document)
        foot_pairs = {
            source_name: target_name
            for source_name, target_name in mapping.required_pairs().items()
            if source_name.endswith("_foot")
        }
        grounded = solve_contacts(
            target, target_motion, source_motion, foot_pairs, tolerance=ground_tolerance
        )
        frames = grounded["frames"]

        bind_global = global_bind_rotations(target)
        first_globals = global_frame_rotations(target, frames[0], bind_global)
        first_yaw = _world_yaw(first_globals[target.root])
        first_root = tuple(frames[0]["root_translation"])
        if clip_index == 0:
            yaw_delta = 0.0
            alignment_root = (0.0, 0.0, 0.0)
        else:
            previous = output_frames[-1]
            previous_globals = global_frame_rotations(target, previous, bind_global)
            yaw_delta = 0.0
            alignment_root = vsub(previous["root_translation"], first_root)
        from .math3d import euler_xyz_to_quat
        yaw_quat = euler_xyz_to_quat((0.0, yaw_delta, 0.0))
        clip_start_root = None
        clip_end_root = None
        for local_index, frame in enumerate(frames):
            global_index = frame_offset + local_index
            root = frame["root_translation"]
            relative_root = vsub(root, first_root)
            rotated = q_rotate_vector(yaw_quat, relative_root)
            new_root = [rotated[i] + alignment_root[i] + first_root[i] for i in range(3)]
            frame["root_translation"] = new_root
            frame["root_yaw_delta"] = first_yaw
            if local_index == 0:
                clip_start_root = list(new_root)
            clip_end_root = list(new_root)
            output_frames.append(frame)

        # Transform contact intervals and anchors into merged coordinates.
        for source_foot, target_foot in foot_pairs.items():
            intervals = source_motion.contacts.get(source_foot, [])
            if not intervals:
                continue
            merged_intervals = output_contacts.setdefault(target_foot, [])
            for interval in intervals:
                start = frame_offset + int(interval["start"])
                end = frame_offset + int(interval["end"])
                merged_intervals.append({"start": start, "end": end})
        for slide in grounded["slides"]:
            merged_slide = dict(slide)
            merged_slide["start"] = frame_offset + slide["start"]
            merged_slide["end"] = frame_offset + slide["end"]
            anchor_relative = vsub(slide["anchor"], first_root)
            merged_slide["anchor"] = [
                anchor_relative[i] + first_root[i] + alignment_root[i]
                for i in range(3)
            ]
            contact_reports.append(merged_slide)

        provenance.append({
            "clip": clip_index,
            "source_resource": clip.get("source_path", source_motion.name),
            "source_motion": source_motion.name,
            "mapping": mapping.name,
            "mapping_version": mapping.version,
            "frame_start": frame_offset,
            "frame_end": frame_offset + len(frames) - 1,
            "local_revisions": deepcopy(clip.get("revisions", [])),
            "root_start": clip_start_root,
            "root_end": clip_end_root,
            "root_displacement": float(vdist(clip_end_root, clip_start_root)),
            "unmet_contacts": [
                {**issue, "frame": issue["frame"] + frame_offset}
                for issue in grounded["unmet"]
            ],
        })
        if clip_index > 0:
            previous_root = output_frames[frame_offset - 1]["root_translation"]
            first_root = output_frames[frame_offset]["root_translation"]
            transitions.append({
                "between_clips": [clip_index - 1, clip_index],
                "frame": frame_offset,
                "root_gap": vdist(previous_root, first_root),
            })
        frame_offset += len(frames)

    return {
        "name": name,
        "kind": "merged_motion",
        "skeleton": target.name,
        "target_skeleton": target.name,
        "frame_rate": frame_rate,
        "root_scale": None,
        "frames": output_frames,
        "contacts": output_contacts,
        "provenance": provenance,
        "merge_metadata": {
            "method": "per-clip bind-relative retarget, contact solve, root transform stitching",
            "mapping_versions": sorted(mapping_versions),
            "source_resources": source_resources,
            "transitions": transitions,
            "contact_slides": contact_reports,
        },
    }
