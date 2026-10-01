from __future__ import annotations

from copy import deepcopy
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Skeleton

from .contacts import LOCK_WEIGHT, contact_weight_from_raw
from .math3d import vlength, vsub
from .retarget import pose_world_positions


def _group_for(contact: dict[str, Any]) -> str:
    if contact.get("support_group"):
        return str(contact["support_group"])
    joint = str(contact.get("joint", ""))
    if joint.endswith("_hand"):
        return "hands"
    if joint.endswith("_foot"):
        return "feet"
    return joint or "body"


def _key(contact: dict[str, Any], index: int) -> str:
    return str(contact.get("contact_id") or contact.get("name") or f"{contact['joint']}_{index}")


def _frames(frames: list[int]) -> list[dict[str, int]]:
    if not frames:
        return []
    result = []
    start = end = frames[0]
    for frame in frames[1:]:
        if frame == end + 1:
            end = frame
        else:
            result.append({"start": start, "end": end})
            start = end = frame
    result.append({"start": start, "end": end})
    return result


def _link(left: dict[str, Any], right: dict[str, Any], left_index: int, right_index: int) -> dict[str, Any]:
    overlap = min(int(left["end"]), int(right["end"])) - max(int(left["start"]), int(right["start"])) + 1
    return {
        "from": _key(left, left_index),
        "to": _key(right, right_index),
        "release_frame": int(left["end"]),
        "acquire_frame": int(right["start"]),
        "overlap_frames": max(0, overlap),
        "unsupported_frames": max(0, int(right["start"]) - int(left["end"]) - 1),
        "support_group": _group_for(right),
    }


def build_handoff_audit(target: Skeleton, motion_data: dict[str, Any], report: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    contacts = motion_data.get("contacts", [])
    unmet: dict[str, list[int]] = {}
    yielded: dict[int, list[dict[str, Any]]] = {}
    for item in report.get("unmet", []):
        unmet.setdefault(str(item.get("contact") or item.get("joint")), []).append(int(item["frame"]))
        yielded.setdefault(int(item["frame"]), []).append({
            "joint": item["joint"],
            "contact": item.get("contact"),
            "error": item.get("error", 0.0),
            "yielded_to": item.get("yielded_to", []),
            "reason": item.get("reason"),
        })

    timeline = []
    for frame in range(len(motion_data["frames"])):
        supports = []
        for index, contact in enumerate(contacts):
            if not int(contact["start"]) <= frame <= int(contact["end"]):
                continue
            weight = contact_weight_from_raw(contact, frame)
            if weight <= 1e-9:
                continue
            key = _key(contact, index)
            supports.append({
                "key": key,
                "joint": contact["joint"],
                "contact": contact.get("name"),
                "support_group": _group_for(contact),
                "weight": weight,
                "locked": weight >= LOCK_WEIGHT,
                "moving": bool(contact.get("velocity") or contact.get("target_path")),
                "status": "yielding" if frame in unmet.get(key, []) else "supporting",
            })
        timeline.append({
            "frame": frame,
            "supports": supports,
            "yielded": yielded.get(frame, []),
            "locked_support_count": sum(1 for item in supports if item["locked"]),
            "support_gap": not any(item["locked"] for item in supports),
        })

    old = motion_data.get("contact_histories") or motion_data.get("contact_handoffs", {}).get("contact_histories", {})
    histories = {}
    for index, contact in enumerate(contacts):
        key = _key(contact, index)
        start, end = int(contact["start"]), int(contact["end"])
        locked = [f for f in range(start, end + 1) if contact_weight_from_raw(contact, f) >= LOCK_WEIGHT]
        current = {
            "joint": contact["joint"],
            "support_group": _group_for(contact),
            "start": start,
            "end": end,
            "moving": bool(contact.get("velocity") or contact.get("target_path")),
            "acquired": locked[0] if locked else start,
            "released": end,
            "handoff_from": contact.get("handoff_from"),
            "handoff_to": contact.get("handoff_to"),
            "yield_ranges": _frames(unmet.get(key, [])),
        }
        previous = old.get(key, {})
        histories[key] = {
            **previous,
            "contact_id": key,
            "current": current,
            "revision_count": int(previous.get("revision_count", 0)) + (0 if previous.get("current") == current else 1),
        }

    ordered = list(enumerate(contacts))
    key_map = {_key(contact, index): (index, contact) for index, contact in ordered}
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, contact in ordered:
        groups.setdefault(_group_for(contact), []).append((index, contact))
    handoffs = []
    for group in groups.values():
        group.sort(key=lambda item: (int(item[1]["start"]), int(item[1]["end"]), item[0]))
        handoffs += [_link(left[1], right[1], left[0], right[0]) for left, right in zip(group, group[1:])]
    existing = {(item["from"], item["to"]) for item in handoffs}
    for index, contact in ordered:
        if contact.get("handoff_from") in key_map:
            other_index, other = key_map[contact["handoff_from"]]
            item = _link(other, contact, other_index, index)
            pair = (item["from"], item["to"])
            if pair not in existing:
                handoffs.append(item)
                existing.add(pair)
        if contact.get("handoff_to") in key_map:
            other_index, other = key_map[contact["handoff_to"]]
            item = _link(contact, other, index, other_index)
            pair = (item["from"], item["to"])
            if pair not in existing:
                handoffs.append(item)
                existing.add(pair)
    handoffs.sort(key=lambda item: (item["acquire_frame"], item["from"], item["to"]))

    moving = []
    for index, contact in enumerate(contacts):
        path = contact.get("anchor_path")
        if not path:
            continue
        displacement = [float(path[-1][axis] - path[0][axis]) for axis in range(3)]
        distance = vlength(displacement)
        moving.append({
            "contact": _key(contact, index),
            "joint": contact["joint"],
            "displacement": displacement,
            "distance": distance,
            "direction": "stationary" if distance <= 1e-9 else ("positive_x" if displacement[0] >= 0 else "negative_x"),
        })

    cyclic = None
    if motion_data.get("cyclic"):
        first = pose_world_positions(target, motion_data["frames"][0])
        last = pose_world_positions(target, motion_data["frames"][-1])
        gaps = {joint: vlength(vsub(last[joint], first[joint])) for joint in target.joints}
        cyclic = {
            "root_gap": vlength(vsub(motion_data["frames"][-1]["root_translation"], motion_data["frames"][0]["root_translation"])),
            "max_joint_gap": max(gaps.values(), default=0.0),
            "joint_gaps": gaps,
        }

    return {
        "solve_start": kwargs.get("solve_start"),
        "manual_pins": list(kwargs.get("manual_pins") or []),
        "timeline": timeline,
        "handoffs": handoffs,
        "contact_histories": histories,
        "moving_support_checks": moving,
        "skeleton_scale": float(motion_data.get("root_scale", 1.0)),
        "cyclic_closure": cyclic,
    }


def attach_handoff_audit(target: Skeleton, motion_data: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    audit = build_handoff_audit(
        target,
        motion_data,
        report,
        solve_start=report.get("solve_start"),
        manual_pins=report.get("manual_pins"),
    )
    motion_data["contact_handoffs"] = audit
    if audit["contact_histories"]:
        motion_data["contact_histories"] = deepcopy(audit["contact_histories"])
    report["handoff_audit"] = audit
    return audit
