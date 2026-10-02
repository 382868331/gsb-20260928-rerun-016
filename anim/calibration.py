from __future__ import annotations

from copy import deepcopy
from typing import Any

from .math3d import q_rotate_vector, quat_from_euler_xyz, vsub, vlength
from .retarget import pose_world_positions


WORLD = "world"
RELATIVE = "relative"


def _as_point(value) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError("calibration point must contain three components")
    return tuple(float(component) for component in value)  # type: ignore[return-value]


def _revision_rotation(revision: dict[str, Any]):
    if "rotation" in revision:
        rotation = revision["rotation"]
        if len(rotation) != 4:
            raise ValueError("calibration rotation must be a quaternion [w, x, y, z]")
        return tuple(float(component) for component in rotation)
    yaw = float(revision.get("rotation_y", revision.get("yaw", 0.0)))
    return quat_from_euler_xyz((0.0, 0.0, yaw))


def _transform_point(point, revision):
    old_origin = _as_point(revision.get("old_origin", (0.0, 0.0, 0.0)))
    new_origin = _as_point(revision.get("new_origin", (0.0, 0.0, 0.0)))
    rotated = q_rotate_vector(_revision_rotation(revision), vsub(point, old_origin))
    return tuple(new_origin[index] + rotated[index] for index in range(3))


def _target_names(revision):
    names = revision.get("targets", revision.get("contacts", []))
    if "target" in revision:
        names = [*names, revision["target"]]
    if "contact" in revision:
        names = [*names, revision["contact"]]
    return {str(name) for name in names}


def _applies_to_contact(contact, revision):
    names = _target_names(revision)
    if not names:
        return True
    return contact.get("name") in names or contact.get("joint") in names


def _baseline_anchor(target, motion_data, contact):
    start_pose = pose_world_positions(target, motion_data["frames"][int(contact["start"])])
    start_position = start_pose[contact["joint"]]
    return (start_position[0], float(contact.get("ground_y", 0.0)) * float(motion_data.get("root_scale", 1.0)), start_position[2])


def _baseline_path(target, motion_data, contact):
    start, end = int(contact["start"]), int(contact["end"])
    raw_path = contact.get("target_path")
    if raw_path is not None:
        scale = float(motion_data.get("root_scale", 1.0))
        return [
            None if point is None else tuple(float(component) * scale for component in point)
            for point in raw_path
        ]
    first = _baseline_anchor(target, motion_data, contact)
    path = [first for _ in range(end - start + 1)]
    velocity = contact.get("velocity")
    if velocity is None:
        return path
    frame_rate = float(motion_data.get("frame_rate", 30.0))
    scale = float(motion_data.get("root_scale", 1.0))
    velocity_scaled = tuple(float(component) * scale for component in velocity)
    return [
        tuple(
            first[component] + velocity_scaled[component] * (offset / max(frame_rate, 1e-9))
            for component in range(3)
        )
        for offset in range(end - start + 1)
    ]


def revision_missing_frames(revision):
    if not revision:
        return []
    missing = revision.get("missing_frames", revision.get("gap_frames", []))
    if isinstance(missing, dict):
        frames = []
        for start, end in missing.get("ranges", []):
            frames.extend(range(int(start), int(end) + 1))
        frames.extend(int(frame) for frame in missing.get("frames", []))
        return sorted(set(frames))
    return [int(frame) for frame in missing]


def apply_calibration_revision(target, motion_data, revision, *, confirmed_before_frame=None):
    """Apply world or support-relative recalibration to a copied document."""
    interpretation = revision.get("interpretation", revision.get("mode", RELATIVE))
    if interpretation not in {WORLD, RELATIVE}:
        raise ValueError(f"unknown calibration interpretation: {interpretation}")
    document = deepcopy(motion_data)
    baseline_contacts = deepcopy(document.get("contacts", []))
    apply_from = int(revision.get("apply_from", revision.get("start", 0)))
    if confirmed_before_frame is not None:
        apply_from = max(apply_from, int(confirmed_before_frame))
    missing = set(revision_missing_frames(revision))
    revision_id = revision.get("name") or revision.get("id") or revision.get("revision")

    for contact in document.get("contacts", []):
        if not _applies_to_contact(contact, revision):
            continue
        start, end = int(contact["start"]), int(contact["end"])
        path = _baseline_path(target, document, contact)
        updated_path = []
        jumps = []
        previous_anchor = None
        for offset, anchor in enumerate(path):
            frame = start + offset
            if frame in missing or anchor is None:
                updated_path.append(None)
                previous_anchor = None
                continue
            new_anchor = _transform_point(anchor, revision) if interpretation == RELATIVE and frame >= apply_from else anchor
            if previous_anchor is not None:
                jump = vlength(vsub(new_anchor, previous_anchor))
                if jump > float(revision.get("continuity_tolerance", 0.005)):
                    jumps.append({"frame": frame, "jump": jump, "from": list(previous_anchor), "to": list(new_anchor)})
            previous_anchor = new_anchor
            updated_path.append(list(new_anchor))
        contact["target_path"] = updated_path
        contact["target_path_space"] = "world"
        contact.pop("velocity", None)
        contact["calibration"] = {
            "revision": revision_id,
            "interpretation": interpretation,
            "apply_from": apply_from,
            "status": revision.get("status", "draft"),
            "basis": revision.get("basis"),
            "missing_frames": sorted(frame for frame in missing if start <= frame <= end),
            "boundary_jumps": jumps,
        }

    document.setdefault("calibration_revisions", []).append(deepcopy(revision))
    document.setdefault("_calibration_baselines", {})[str(revision_id)] = baseline_contacts
    document["calibration_audit"] = calibration_context(document, revision, apply_from=apply_from)
    return document


def calibration_context(document, revision, *, apply_from=None):
    frame_count = len(document.get("frames", []))
    apply_from = int(apply_from if apply_from is not None else revision.get("apply_from", revision.get("start", 0)))
    first_root = document["frames"][0]["root_translation"] if frame_count else [0.0, 0.0, 0.0]
    last_root = document["frames"][-1]["root_translation"] if frame_count else first_root
    loop_seam = None
    loop_contact_gaps = []
    if document.get("loop"):
        loop_seam = vlength(vsub(tuple(last_root), tuple(first_root)))
        for contact in document.get("contacts", []):
            path = contact.get("target_path", [])
            if int(contact.get("start", -1)) == 0 and int(contact.get("end", -2)) == frame_count - 1 and path and path[0] is not None and path[-1] is not None:
                loop_contact_gaps.append({
                    "joint": contact.get("joint"),
                    "contact": contact.get("name"),
                    "gap": vlength(vsub(tuple(path[-1]), tuple(path[0]))),
                })
    return {
        "revision": revision.get("name") or revision.get("id") or revision.get("revision"),
        "interpretation": revision.get("interpretation", revision.get("mode", RELATIVE)),
        "apply_from": apply_from,
        "confirmed_before_frame": apply_from,
        "frame_count": frame_count,
        "frame_rate": document.get("frame_rate"),
        "source_time_scale": float(revision.get("source_time_scale", revision.get("time_scale", 1.0))),
        "root_scale": document.get("root_scale"),
        "skeleton_scale": document.get("root_scale"),
        "target_skeleton": document.get("target_skeleton") or document.get("skeleton"),
        "missing_frames": revision_missing_frames(revision),
        "source_time_policy": "frame_rate and source timing are rechecked without implicit time scaling",
        "skeleton_policy": "existing mapping/root_scale is retained and explicitly recorded",
        "loop_seam_root_gap": loop_seam,
        "loop_seam_contact_gaps": loop_contact_gaps,
        "manual_choices": deepcopy(revision.get("manual_choices", [])),
    }


def publish_calibration(document, revision_name, *, evidence=None):
    published = deepcopy(document)
    selected = None
    for revision in published.get("calibration_revisions", []):
        name = revision.get("name") or revision.get("id") or revision.get("revision")
        if name == revision_name:
            revision["status"] = "published"
            selected = revision
    if selected is None:
        raise ValueError(f"unknown calibration revision: {revision_name}")
    snapshot = {
        "revision": revision_name,
        "calibration": deepcopy(selected),
        "contact_basis": deepcopy(published.get("contacts", [])),
        "manual_pins": deepcopy(published.get("manual_pins", [])),
        "contact_history": deepcopy(published.get("contact_history", [])),
        "grounding": deepcopy(published.get("grounding", {})),
        "evidence": deepcopy(evidence or {}),
    }
    published.setdefault("calibration_publications", []).append(snapshot)
    published["published_calibration"] = revision_name
    return published


def withdraw_draft_calibration(document, revision_name):
    updated = deepcopy(document)
    remaining = []
    removed = None
    for revision in updated.get("calibration_revisions", []):
        name = revision.get("name") or revision.get("id") or revision.get("revision")
        if name == revision_name:
            if revision.get("status", "draft") != "draft":
                raise ValueError(f"published calibration cannot be withdrawn: {revision_name}")
            removed = revision
            continue
        remaining.append(revision)
    if removed is None:
        raise ValueError(f"unknown draft calibration: {revision_name}")
    updated["calibration_revisions"] = remaining
    baseline = updated.get("_calibration_baselines", {}).get(revision_name)
    if baseline is not None:
        updated["contacts"] = deepcopy(baseline)
    updated["withdrawn_calibrations"] = [*updated.get("withdrawn_calibrations", []), revision_name]
    return updated
