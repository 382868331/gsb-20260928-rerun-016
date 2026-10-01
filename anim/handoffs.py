from __future__ import annotations

from typing import Any


def _frame_supporters(intervals: list[dict[str, Any]], frame: int) -> list[dict[str, Any]]:
    supporters = []
    for interval in intervals:
        if interval["start"] <= frame <= interval["end"]:
            weight = float(interval.get("weight", 1.0))
            if weight > 1e-9:
                supporters.append({
                    "joint": interval["joint"],
                    "target": interval.get("contact") or interval["joint"],
                    "weight": weight,
                    "moving": bool(interval.get("moving", False)),
                    "error": float(interval.get("error", 0.0)),
                    "supporting": weight * max(0.0, 1.0 - float(interval.get("error", 0.0)) / 0.05),
                })
    return sorted(supporters, key=lambda item: (-item["supporting"], item["target"]))


def _segment_frames(frames: list[int]) -> list[dict[str, int]]:
    if not frames:
        return []
    segments = []
    start = previous = frames[0]
    for frame in frames[1:]:
        if frame == previous + 1:
            previous = frame
            continue
        segments.append({"start": start, "end": previous})
        start = previous = frame
    segments.append({"start": start, "end": previous})
    return segments


def build_handoff_report(
    intervals: list[dict[str, Any]],
    frame_count: int,
    *, unmet: list[dict[str, Any]] | None = None,
    handoff_order: list[str] | None = None,
) -> dict[str, Any]:
    timeline = []
    support_gap_frames: list[int] = []
    for frame in range(frame_count):
        supporters = _frame_supporters(intervals, frame)
        if not supporters:
            support_gap_frames.append(frame)
        timeline.append({
            "frame": frame,
            "supporters": supporters,
            "state": (
                "overlap" if len(supporters) > 1 else
                "unsupported" if not supporters else "supported"
            ),
        })

    events = []
    ordered = sorted(intervals, key=lambda item: (item["start"], item["end"], item.get("contact") or item["joint"]))
    for index, interval in enumerate(ordered):
        key = interval.get("contact") or interval["joint"]
        events.append({
            "type": "engage",
            "frame": interval["start"],
            "joint": interval["joint"],
            "from_target": None,
            "to_target": key,
        })
        events.append({
            "type": "release",
            "frame": interval["end"] + 1,
            "joint": interval["joint"],
            "from_target": key,
            "to_target": None,
        })
        for other in ordered[index + 1:]:
            other_key = other.get("contact") or other["joint"]
            overlap_start = max(interval["start"], other["start"])
            overlap_end = min(interval["end"], other["end"])
            if overlap_start <= overlap_end:
                events.append({
                    "type": "overlap",
                    "frame": overlap_start,
                    "joints": sorted({interval["joint"], other["joint"]}),
                    "from_target": key,
                    "to_target": other_key,
                    "end_frame": overlap_end,
                })

    for item in unmet or []:
        if item.get("yielded"):
            events.append({
                "type": "yield",
                "frame": item["frame"],
                "joint": item["joint"],
                "from_target": item.get("contact"),
                "to_targets": item.get("yielded_to", []),
                "error": item.get("error", 0.0),
            })

    history = []
    for interval in ordered:
        key = interval.get("contact") or interval["joint"]
        overlapping = []
        for other in ordered:
            if other is interval:
                continue
            if max(interval["start"], other["start"]) <= min(interval["end"], other["end"]):
                overlapping.append(other.get("contact") or other["joint"])
        history.append({
            "target": key,
            "joint": interval["joint"],
            "start": interval["start"],
            "end": interval["end"],
            "moving": bool(interval.get("moving", False)),
            "overlaps_with": sorted(set(overlapping)),
            "met": bool(interval.get("met", True)),
            "max_error": float(interval.get("max_error", 0.0)),
        })

    events.sort(key=lambda event: (event["frame"], event["type"]))
    return {
        "handoff_order": handoff_order or [item.get("contact") or item["joint"] for item in ordered],
        "timeline": timeline,
        "events": events,
        "support_gaps": _segment_frames(support_gap_frames),
        "contact_history": history,
    }
