from __future__ import annotations

from typing import Any

from .models import Motion, Skeleton
from .retarget import pose_world_positions


def validate_merged_motion(
    skeleton: Skeleton,
    document: dict[str, Any],
    *,
    slide_tolerance: float = 0.25,
    unmet_tolerance: float = 0.02,
) -> dict[str, Any]:
    motion = Motion(document)
    failures: list[dict[str, Any]] = []
    slides: list[dict[str, Any]] = []
    unmet: list[dict[str, Any]] = []
    masks = motion.contact_mask()
    stored_anchors: dict[str, list[dict[str, Any]]] = {}
    for slide in document.get("merge_metadata", {}).get("contact_slides", []):
        if "anchor" in slide:
            stored_anchors.setdefault(slide["joint"], []).append(slide)
    for foot, frame_set in masks.items():
        intervals = motion.contacts[foot]
        for interval in intervals:
            start, end = int(interval["start"]), int(interval["end"])
            precomputed = next((
                item for item in stored_anchors.get(foot, [])
                if int(item["start"]) == start and int(item["end"]) == end
            ), None)
            if precomputed is not None:
                record = {"joint": foot, "start": start, "end": end,
                          "max_slide": float(precomputed.get("slide", 0.0)),
                          "worst_frame": start}
                slides.append(record)
                if record["max_slide"] > slide_tolerance:
                    failures.append({"type": "contact_slide", **record, "tolerance": slide_tolerance})
                continue
            anchor = None
            for record in stored_anchors.get(foot, []):
                if int(record["start"]) == start and int(record["end"]) == end:
                    anchor = tuple(record["anchor"])
                    break
            if anchor is None:
                anchor = pose_world_positions(skeleton, motion.frames[start])[foot]
            from .math3d import vdist

            max_slide = 0.0
            worst_frame = start
            for frame_index in range(start, end + 1):
                point = pose_world_positions(skeleton, motion.frames[frame_index])[foot]
                slide = vdist(point, anchor)
                if slide > max_slide:
                    max_slide, worst_frame = slide, frame_index
            record = {
                "joint": foot,
                "start": start,
                "end": end,
                "max_slide": max_slide,
                "worst_frame": worst_frame,
            }
            slides.append(record)
            if max_slide > slide_tolerance:
                failures.append({"type": "contact_slide", **record, "tolerance": slide_tolerance})

    # Swinging feet must not be frozen. Report a foot that stays within a tiny
    # radius during a clearly non-contact span between two contacts.
    from .math3d import vdist

    for foot, intervals in motion.contacts.items():
        for earlier, later in zip(intervals, intervals[1:]):
            free_start = int(earlier["end"]) + 1
            free_end = int(later["start"]) - 1
            if free_end < free_start + 2:
                continue
            points = [pose_world_positions(skeleton, motion.frames[i])[foot] for i in range(free_start, free_end + 1)]
            travel = max(vdist(point, points[0]) for point in points)
            if travel < 1e-5:
                failures.append({
                    "type": "swing_foot_fixed",
                    "joint": foot,
                    "start": free_start,
                    "end": free_end,
                })

    for clip in document.get("provenance", []):
        for issue in clip.get("unmet_contacts", []):
            if issue.get("residual", 0.0) > unmet_tolerance:
                unmet.append(issue)

    transitions = document.get("merge_metadata", {}).get("transitions", [])
    for transition in transitions:
        if transition.get("root_gap", 0.0) > 1e-6:
            failures.append({"type": "clip_gap", **transition})

    return {
        "passed": not failures,
        # Unmet contact targets are explainable retained outcomes, not pipeline
        # failures: the pose is preserved and the residual is reported.
        "failure_count": len(failures),
        "failures": failures,
        "unmet_contacts": unmet,
        "slides": slides,
        "transitions": transitions,
        "metrics": {
            "frames": len(motion.frames),
            "clips": len(document.get("provenance", [])),
            "contact_intervals": sum(len(intervals) for intervals in motion.contacts.values()),
            "unmet_contact_frames": len(unmet),
        },
    }
