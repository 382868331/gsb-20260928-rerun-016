from __future__ import annotations

from copy import deepcopy
from typing import Any

from .math3d import q_rotate_vector, quat_from_euler_xyz, vsub, vlength
from .retarget import pose_world_positions


WORLD = "world"
RELATIVE = "relative"


def revision_id(revision: dict[str, Any] | None):
    if not revision:
        return None
    value = revision.get("name", revision.get("id", revision.get("revision")))
    return str(value) if value is not None else None


def revision_range(revision: dict[str, Any]) -> tuple[int, int | None]:
    start = int(revision.get("apply_from", revision.get("start", 0)))
    end_value = revision.get("apply_until", revision.get("end"))
    end = int(end_value) if end_value is not None else None
    return start, end


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
        path_space = contact.get("target_path_space", "resource")
        return [
            None if point is None else tuple(
                float(component) * (scale if path_space == "resource" else 1.0)
                for component in point
            )
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


def _stored_baseline_path(document, contact):
    key = str(contact.get("name") or contact.get("joint"))
    paths = document.get("_calibration_contact_baseline_paths", {})
    if key not in paths:
        return None
    return [None if point is None else tuple(point) for point in paths[key]]


def _original_contacts(document: dict[str, Any]):
    baseline = document.get("_calibration_contact_baseline")
    if baseline is not None:
        return deepcopy(baseline)
    stored = document.get("_calibration_baselines", {})
    if stored:
        first_key = sorted(stored)[0]
        return deepcopy(stored[first_key])
    return deepcopy(document.get("contacts", []))


def _normalise_decision(decision: Any) -> dict[str, Any]:
    if not isinstance(decision, dict):
        revision, interpretation = decision
        decision = {"revision": revision, "interpretation": interpretation}
    result = deepcopy(decision)
    revision = result.get("revision", result.get("name", result.get("id")))
    if revision is None:
        raise ValueError("calibration decision must identify a revision")
    result["revision"] = str(revision)
    result["interpretation"] = result.get("interpretation", result.get("mode", RELATIVE))
    if result["interpretation"] not in {WORLD, RELATIVE}:
        raise ValueError(f"unknown calibration interpretation: {result['interpretation']}")
    if "start" in result and "frames" not in result:
        result["start"] = int(result["start"])
        result["end"] = int(result.get("end", result["start"]))
    return result


def _decision_matches(decision: dict[str, Any], revision_id_value: str, frame: int) -> bool:
    if decision["revision"] != revision_id_value:
        return False
    if "frames" in decision:
        return int(frame) in {int(value) for value in decision["frames"]}
    if "ranges" in decision:
        return any(int(start) <= frame <= int(end) for start, end in decision["ranges"])
    start = int(decision.get("start", decision.get("apply_from", 0)))
    end_value = decision.get("end", decision.get("apply_until"))
    end = int(end_value) if end_value is not None else None
    return frame >= start and (end is None or frame <= end)


def _decision_for_frame(decisions, revision_id_value: str, frame: int):
    matches = [decision for decision in decisions if _decision_matches(decision, revision_id_value, frame)]
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"overlapping decisions for calibration {revision_id_value} at frame {frame}")
    return matches[0]


def _revision_at_frame(revision: dict[str, Any], frame: int) -> bool:
    start, end = revision_range(revision)
    return frame >= start and (end is None or frame <= end)


def _candidate_at_frame(revision, baseline_anchor, frame, decision):
    interpretation = decision.get("interpretation", revision.get("interpretation", revision.get("mode", RELATIVE)))
    if interpretation not in {WORLD, RELATIVE}:
        raise ValueError(f"unknown calibration interpretation: {interpretation}")
    transformed = _transform_point(baseline_anchor, revision) if interpretation == RELATIVE else baseline_anchor
    return {
        "revision": revision_id(revision),
        "interpretation": interpretation,
        "position": list(transformed),
        "basis": decision.get("basis", revision.get("basis")),
        "decision": decision.get("decision", decision.get("name")),
    }


def _conflict_segments(contact, revisions, decisions):
    start, end = int(contact["start"]), int(contact["end"])
    segments = []
    current = None
    for frame in range(start, end + 1):
        candidates = []
        for revision in revisions:
            if not _applies_to_contact(contact, revision) or not _revision_at_frame(revision, frame):
                continue
            rejected_at_frame = False
            for decision in decisions:
                rejected_ids = {str(value) for value in decision.get("rejected_revisions", decision.get("alternative_revisions", []))}
                if revision_id(revision) in rejected_ids and _decision_matches(decision, decision["revision"], frame):
                    rejected_at_frame = True
                    break
            if rejected_at_frame:
                continue
            decision = _decision_for_frame(decisions, revision_id(revision), frame)
            if decision is None:
                candidates.append({
                    "revision": revision_id(revision),
                    "interpretations": [WORLD, RELATIVE],
                    "basis": revision.get("basis"),
                })
            else:
                candidates.append({
                    "revision": revision_id(revision),
                    "interpretation": decision.get("interpretation"),
                    "interpretations": [decision.get("interpretation")],
                    "decision": decision.get("decision", decision.get("name")),
                    "basis": decision.get("basis", revision.get("basis")),
                })
        signature = tuple((item["revision"], tuple(item.get("interpretations", [item.get("interpretation")])), item.get("decision")) for item in candidates)
        if current is not None and signature == current["signature"]:
            current["end"] = frame
            current["frames"].append(frame)
        else:
            current = {
                "start": frame,
                "end": frame,
                "frames": [frame],
                "signature": signature,
                "candidates": candidates,
            }
            segments.append(current)
    unresolved = [
        {
            "start": segment["start"],
            "end": segment["end"],
            "frames": segment["frames"],
            "candidates": [
                {key: value for key, value in candidate.items() if key != "decision"}
                for candidate in segment["candidates"]
            ],
        }
        for segment in segments
        if len(segment["candidates"]) > 1
    ]
    return unresolved, [segment for segment in segments if len(segment["candidates"]) <= 1]


def recompute_calibrated_contacts(target, motion_data, revisions, decisions=None):
    document = deepcopy(motion_data)
    revisions = sorted((deepcopy(revision) for revision in revisions), key=revision_id)
    decisions = [_normalise_decision(decision) for decision in (decisions or document.get("calibration_decisions", []))]
    known = {revision_id(revision) for revision in revisions}
    for decision in decisions:
        if decision["revision"] not in known:
            raise ValueError(f"unknown calibration decision revision: {decision['revision']}")

    contacts = _original_contacts(document)
    baseline_paths = {}
    unresolved_conflicts = []
    for contact in contacts:
        start, end = int(contact["start"]), int(contact["end"])
        contact_revisions = [revision for revision in revisions if _applies_to_contact(contact, revision)]
        frozen_before = min((revision_range(revision)[0] for revision in contact_revisions), default=0)
        conflicts, resolved_segments = _conflict_segments(contact, contact_revisions, decisions)
        if conflicts:
            unresolved_conflicts.append({
                "contact": contact.get("name"),
                "joint": contact.get("joint"),
                "segments": conflicts,
            })
        baseline_path = _stored_baseline_path(document, contact)
        if baseline_path is None:
            if target is None:
                raise ValueError("target skeleton is required when no stored calibration baseline path exists")
            baseline_path = _baseline_path(target, document, contact)
        baseline_paths[str(contact.get("name") or contact.get("joint"))] = [list(point) if point is not None else None for point in baseline_path]
        all_missing = set().union(*(set(revision_missing_frames(revision)) for revision in contact_revisions)) if contact_revisions else set()
        unresolved_lookup = {
            frame: segment
            for segment in conflicts
            for frame in segment["frames"]
        }
        updated_path = []
        jumps = []
        previous_anchor = None
        selected_frames = []
        segment_basis = []
        active_basis_key = object()
        for offset, baseline_anchor in enumerate(baseline_path):
            frame = start + offset
            current_segment = next((segment for segment in resolved_segments if segment["start"] <= frame <= segment["end"] and segment["candidates"]), None)
            selected = None
            if current_segment is not None:
                candidate = current_segment["candidates"][0]
                revision = next(item for item in contact_revisions if revision_id(item) == candidate["revision"])
                decision = _decision_for_frame(decisions, candidate["revision"], frame) or {}
                selected = (revision, decision)

            active_missing = False
            if current_segment is not None:
                active_missing = frame in set(revision_missing_frames(selected[0]))
            elif frame in unresolved_lookup:
                active_ids = {candidate["revision"] for candidate in unresolved_lookup[frame]["candidates"]}
                active_missing = any(
                    frame in set(revision_missing_frames(revision))
                    for revision in contact_revisions
                    if revision_id(revision) in active_ids
                )
            if active_missing or baseline_anchor is None:
                updated_path.append(None)
                previous_anchor = None
                active_basis_key = object()
                continue
            if selected is None:
                new_anchor = baseline_anchor
                selected_meta = {
                    "revision": None,
                    "interpretation": "frozen" if frame < frozen_before else WORLD,
                    "basis": "confirmed prefix" if frame < frozen_before else "unchanged contact basis",
                }
                basis_key = (selected_meta["revision"], selected_meta["interpretation"], selected_meta["basis"])
                if basis_key != active_basis_key:
                    segment_basis.append({"start": frame, "end": frame, **selected_meta})
                    active_basis_key = basis_key
                else:
                    segment_basis[-1]["end"] = frame
            else:
                revision, decision = selected
                new_anchor = tuple(_candidate_at_frame(revision, baseline_anchor, frame, decision)["position"])
                selected_meta = {
                    "revision": revision_id(revision),
                    "interpretation": decision.get("interpretation", revision.get("interpretation", revision.get("mode", RELATIVE))),
                    "basis": decision.get("basis", revision.get("basis")),
                }
                selected_frames.append(frame)
                basis_key = (selected_meta["revision"], selected_meta["interpretation"], selected_meta["basis"])
                if basis_key != active_basis_key:
                    segment_basis.append({"start": frame, "end": frame, **selected_meta})
                    active_basis_key = basis_key
                else:
                    segment_basis[-1]["end"] = frame
            if previous_anchor is not None:
                jump = vlength(vsub(new_anchor, previous_anchor))
                tolerance_candidates = [float(revision.get("continuity_tolerance", 0.005)) for revision in contact_revisions]
                if jump > min(tolerance_candidates, default=0.005):
                    jumps.append({"frame": frame, "jump": jump, "from": list(previous_anchor), "to": list(new_anchor)})
            previous_anchor = new_anchor
            updated_path.append(list(new_anchor))

        if contact_revisions:
            contact["target_path"] = updated_path
            contact["target_path_space"] = "world"
            contact.pop("velocity", None)
            selected_revision_ids = sorted(revision_id(revision) for revision in contact_revisions)
            contact["calibration"] = {
                "revision": selected_revision_ids[-1] if len(selected_revision_ids) == 1 else None,
                "revisions": selected_revision_ids,
                "interpretation": None,
                "apply_from": min((revision_range(revision)[0] for revision in contact_revisions), default=0),
                "frozen_before": frozen_before,
                "status": "unresolved" if conflicts else "draft",
                "basis": [decision.get("basis") for decision in decisions if decision.get("basis")],
                "missing_frames": sorted(frame for frame in all_missing if start <= frame <= end),
                "boundary_jumps": jumps,
                "segment_basis": segment_basis,
                "unresolved_segments": conflicts,
            }
            if len(contact_revisions) == 1 and not decisions:
                revision = contact_revisions[0]
                contact["calibration"].update({
                    "revision": revision_id(revision),
                    "interpretation": revision.get("interpretation", revision.get("mode", RELATIVE)),
                })
            for segment in conflicts:
                anchor = baseline_path[segment["start"] - start]
                segment["consequences"] = {}
                if anchor is not None and segment["start"] not in all_missing:
                    for candidate in segment["candidates"]:
                        revision = next(item for item in contact_revisions if revision_id(item) == candidate["revision"])
                        candidate_missing = segment["start"] in set(revision_missing_frames(revision))
                        segment["consequences"][candidate["revision"]] = {
                            "missing": candidate_missing,
                            WORLD: None if candidate_missing else list(anchor),
                            RELATIVE: None if candidate_missing else list(_transform_point(anchor, revision)),
                        }
    document["contacts"] = contacts
    document["calibration_revisions"] = deepcopy(revisions)
    document["calibration_decisions"] = deepcopy(decisions)
    document["calibration_conflicts"] = unresolved_conflicts
    document.setdefault("_calibration_contact_baseline", _original_contacts(motion_data))
    document["_calibration_contact_baseline_paths"] = baseline_paths
    document["calibration_audit"] = calibration_audit(document, revisions, decisions, unresolved_conflicts)
    return document


def calibration_audit(document, revisions, decisions, unresolved_conflicts):
    if len(revisions) == 1 and not decisions:
        audit = calibration_context(document, revisions[0])
    else:
        frame_count = len(document.get("frames", []))
        first_root = document["frames"][0]["root_translation"] if frame_count else [0.0, 0.0, 0.0]
        last_root = document["frames"][-1]["root_translation"] if frame_count else first_root
        missing = sorted({frame for revision in revisions for frame in revision_missing_frames(revision)})
        audit = {
            "revision": None,
            "revisions": [revision_id(revision) for revision in revisions],
            "interpretation": None,
            "apply_from": min((revision_range(revision)[0] for revision in revisions), default=0),
            "confirmed_before_frame": min((revision_range(revision)[0] for revision in revisions), default=0),
            "frame_count": frame_count,
            "frame_rate": document.get("frame_rate"),
            "source_time_scale": 1.0,
            "root_scale": document.get("root_scale"),
            "skeleton_scale": document.get("root_scale"),
            "target_skeleton": document.get("target_skeleton") or document.get("skeleton"),
            "missing_frames": missing,
            "source_time_policy": "frame_rate and source timing are rechecked without implicit time scaling",
            "skeleton_policy": "existing mapping/root_scale is retained and explicitly recorded",
            "loop_seam_root_gap": vlength(vsub(tuple(last_root), tuple(first_root))) if document.get("loop") else None,
            "manual_choices": [choice for revision in revisions for choice in deepcopy(revision.get("manual_choices", []))],
        }
    audit.update({
        "decisions": deepcopy(decisions),
        "unresolved_conflicts": deepcopy(unresolved_conflicts),
        "resolution_policy": "overlapping revisions are presented side by side and require explicit per-segment decisions; registration order is not used",
    })
    return audit


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
    candidate = deepcopy(revision)
    apply_from = int(revision.get("apply_from", revision.get("start", 0)))
    if confirmed_before_frame is not None:
        apply_from = max(apply_from, int(confirmed_before_frame))
    candidate["apply_from"] = apply_from
    candidate.setdefault("interpretation", interpretation)
    return register_calibration_revision(target, motion_data, candidate)


def register_calibration_revision(target, motion_data, revision):
    document = deepcopy(motion_data)
    revisions = deepcopy(document.get("calibration_revisions", []))
    new_revision = deepcopy(revision)
    new_revision.setdefault("status", "draft")
    new_id = revision_id(new_revision)
    if new_id is None:
        raise ValueError("calibration revision requires a name, id, or revision")
    if any(revision_id(existing) == new_id for existing in revisions):
        raise ValueError(f"calibration revision already exists: {new_id}")
    document.setdefault("_calibration_contact_baseline", deepcopy(document.get("contacts", [])))
    revisions.append(new_revision)
    decisions = deepcopy(document.get("calibration_decisions", []))
    result = recompute_calibrated_contacts(target, document, revisions, decisions)
    result.setdefault("_calibration_baselines", {})[new_id] = deepcopy(result["_calibration_contact_baseline"])
    return result


def choose_calibration_interpretation(
    target,
    document,
    revision_name,
    interpretation,
    *,
    start=None,
    end=None,
    basis=None,
    decision=None,
    rejected_revisions=None,
):
    revisions = deepcopy(document.get("calibration_revisions", []))
    revision = next((item for item in revisions if revision_id(item) == str(revision_name)), None)
    if revision is None:
        raise ValueError(f"unknown calibration revision: {revision_name}")
    if interpretation not in {WORLD, RELATIVE}:
        raise ValueError(f"unknown calibration interpretation: {interpretation}")
    if start is None:
        start, apply_end = revision_range(revision)
        end = apply_end if end is None else end
    chosen = {
        "revision": str(revision_name),
        "interpretation": interpretation,
        "start": int(start),
        "end": int(end) if end is not None else None,
    }
    if basis is not None:
        chosen["basis"] = basis
    if decision is not None:
        chosen["decision"] = decision
    if rejected_revisions is not None:
        chosen["rejected_revisions"] = [str(item) for item in rejected_revisions]
    replacement_end = int(end) if end is not None else 10**30
    decisions = []
    for existing in deepcopy(document.get("calibration_decisions", [])):
        if existing.get("revision") != str(revision_name):
            decisions.append(existing)
            continue
        overlaps = False
        if "frames" in existing:
            overlaps = any(int(start) <= int(frame) <= replacement_end for frame in existing["frames"])
        elif "ranges" in existing:
            overlaps = any(int(start) <= int(range_end) and int(range_start) <= replacement_end for range_start, range_end in existing["ranges"])
        else:
            existing_start = int(existing.get("start", existing.get("apply_from", 0)))
            existing_end_value = existing.get("end", existing.get("apply_until"))
            existing_end = int(existing_end_value) if existing_end_value is not None else 10**30
            overlaps = int(start) <= existing_end and existing_start <= replacement_end
        if not overlaps:
            decisions.append(existing)
    decisions.append(chosen)
    return recompute_calibrated_contacts(target, document, revisions, decisions)


def choose_calibration_decisions(target, document, decisions):
    return recompute_calibrated_contacts(
        target,
        document,
        deepcopy(document.get("calibration_revisions", [])),
        deepcopy(decisions),
    )


def partial_recalculate_calibration(target, document):
    return recompute_calibrated_contacts(
        target,
        document,
        deepcopy(document.get("calibration_revisions", [])),
        deepcopy(document.get("calibration_decisions", [])),
    )


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
    updated["calibration_decisions"] = [
        decision for decision in updated.get("calibration_decisions", [])
        if decision.get("revision") != revision_name
    ]
    updated.pop("calibration_conflicts", None)
    if remaining:
        updated = recompute_calibrated_contacts(None, updated, remaining, updated.get("calibration_decisions", []))
    else:
        baseline = updated.get("_calibration_contact_baseline")
        if baseline is not None:
            updated["contacts"] = deepcopy(baseline)
        updated.pop("_calibration_contact_baseline", None)
        updated.pop("_calibration_contact_baseline_paths", None)
        updated.pop("calibration_decisions", None)
        updated["calibration_conflicts"] = []
        updated["calibration_audit"] = None
    updated["withdrawn_calibrations"] = [*updated.get("withdrawn_calibrations", []), revision_name]
    return updated
