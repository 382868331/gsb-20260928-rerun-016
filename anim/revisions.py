from __future__ import annotations

from copy import deepcopy
from typing import Any

from .calibration import (
    RELATIVE,
    WORLD,
    _applies_to_contact,
    _baseline_path,
    _transform_point,
    revision_missing_frames,
)
from .math3d import vlength, vsub


def _revision_name(revision: dict[str, Any]) -> str | None:
    name = revision.get("name") or revision.get("id") or revision.get("revision")
    return None if name is None else str(name)


def _revision_interpretation(revision: dict[str, Any]) -> str:
    return revision.get("interpretation", revision.get("mode", RELATIVE))


def _revision_range(revision: dict[str, Any], frame_count: int) -> tuple[int, int]:
    start = int(revision.get("apply_from", revision.get("start", 0)))
    end = revision.get("apply_to", revision.get("end"))
    end = frame_count - 1 if end is None else int(end)
    return start, min(end, frame_count - 1)


def _baseline_contacts(document: dict[str, Any]) -> list[dict[str, Any]]:
    return document.get("_pending_baseline_contacts") or document.get("contacts", [])


def _consequence(target, document, contact, revision, frame):
    path = _baseline_path(target, document, contact)
    offset = frame - int(contact["start"])
    if offset < 0 or offset >= len(path) or path[offset] is None:
        return {"anchor_shift": None, "note": "no recorded anchor at segment start (observation gap)"}
    anchor = path[offset]
    relative_anchor = _transform_point(anchor, revision)
    interpretation = _revision_interpretation(revision)
    return {
        "world_anchor": [float(component) for component in anchor],
        "relative_anchor": [float(component) for component in relative_anchor],
        "anchor_shift": (
            vlength(vsub(relative_anchor, anchor)) if interpretation == RELATIVE else 0.0
        ),
    }


def detect_revision_conflicts(document: dict[str, Any], target=None) -> list[dict[str, Any]]:
    """List frame segments where two or more draft revisions overlap.

    Overlaps are reported side by side with both interpretations and their
    consequences; registration order never decides the outcome.
    """
    pending = [
        revision
        for revision in document.get("pending_revisions", [])
        if revision.get("status", "draft") == "draft"
    ]
    frame_count = len(document.get("frames", []))
    conflicts: list[dict[str, Any]] = []
    for contact in _baseline_contacts(document):
        covering = [revision for revision in pending if _applies_to_contact(contact, revision)]
        if len(covering) < 2:
            continue
        boundaries: set[int] = set()
        ranges: dict[str, tuple[int, int]] = {}
        for revision in covering:
            start, end = _revision_range(revision, frame_count)
            start = max(start, int(contact["start"]))
            end = min(end, int(contact["end"]))
            if start > end:
                continue
            ranges[_revision_name(revision)] = (start, end)
            boundaries.add(start)
            boundaries.add(end + 1)
        ordered = sorted(boundaries)
        for low, high in zip(ordered, ordered[1:]):
            seg_start, seg_end = low, high - 1
            names = [
                name
                for name, (start, end) in ranges.items()
                if start <= seg_start and seg_end <= end
            ]
            if len(names) < 2:
                continue
            options = []
            for name in names:
                revision = next(item for item in covering if _revision_name(item) == name)
                options.append(
                    {
                        "revision": name,
                        "interpretation": _revision_interpretation(revision),
                        "basis": revision.get("basis"),
                        "consequence": (
                            _consequence(target, document, contact, revision, seg_start)
                            if target is not None
                            else None
                        ),
                    }
                )
            conflicts.append(
                {
                    "target": contact.get("name") or contact["joint"],
                    "joint": contact["joint"],
                    "start": seg_start,
                    "end": seg_end,
                    "status": "pending_choice",
                    "options": options,
                }
            )
    return conflicts


def register_pending_revision(document: dict[str, Any], revision: dict[str, Any], target=None) -> dict[str, Any]:
    """Register a draft calibration revision without touching any contact.

    The revision only joins the pending set; overlapping ranges surface as
    conflicts that wait for an explicit per-segment choice.
    """
    name = _revision_name(revision)
    if not name:
        raise ValueError("pending calibration revision requires a name")
    if _revision_interpretation(revision) not in {WORLD, RELATIVE}:
        raise ValueError(f"unknown calibration interpretation: {_revision_interpretation(revision)}")
    updated = deepcopy(document)
    pending = updated.setdefault("pending_revisions", [])
    if any(_revision_name(item) == name for item in pending):
        raise ValueError(f"duplicate pending calibration revision: {name}")
    entry = deepcopy(revision)
    entry.setdefault("status", "draft")
    pending.append(entry)
    updated.setdefault("_pending_baseline_contacts", deepcopy(updated.get("contacts", [])))
    updated["revision_conflicts"] = detect_revision_conflicts(updated, target)
    return updated


def _choice_for(choices, revision_name, frame):
    for choice in choices:
        if choice["revision"] != revision_name:
            continue
        start, end = choice["segment"]
        if int(start) <= frame <= int(end):
            return choice
    return None


def choose_segment_interpretation(
    document: dict[str, Any],
    revision_name: str,
    start: int,
    end: int,
    interpretation: str | None = None,
    *,
    basis: str | None = None,
) -> dict[str, Any]:
    """Record the user's explicit world/relative choice for one time segment."""
    updated = deepcopy(document)
    revision = next(
        (item for item in updated.get("pending_revisions", []) if _revision_name(item) == revision_name),
        None,
    )
    if revision is None:
        raise ValueError(f"unknown pending calibration revision: {revision_name}")
    interpretation = interpretation or _revision_interpretation(revision)
    if interpretation not in {WORLD, RELATIVE}:
        raise ValueError(f"unknown calibration interpretation: {interpretation}")
    choices = updated.setdefault("revision_choices", [])
    choices[:] = [
        choice
        for choice in choices
        if not (choice["revision"] == revision_name and choice["segment"] == [int(start), int(end)])
    ]
    choices.append(
        {
            "revision": revision_name,
            "segment": [int(start), int(end)],
            "interpretation": interpretation,
            "basis": basis if basis is not None else revision.get("basis"),
        }
    )
    return updated


def apply_pending_revisions(
    target,
    document: dict[str, Any],
    *,
    confirmed_before_frame: int | None = None,
    freeze_before_frame: int | None = None,
) -> dict[str, Any]:
    """Apply pending revisions segment by segment from the pristine baseline.

    Every frame covered by more than one draft revision requires an explicit
    choice; unresolved overlaps raise instead of letting registration order
    win.  Missing frames stay empty, frozen prefixes keep their solved
    anchors, and each applied segment records its revision, interpretation
    and basis so the result stays traceable.  Because application always
    restarts from the baseline snapshot, adoption and withdrawal order does
    not change the outcome, and a local recompute from freeze_before_frame
    matches a full recompute under the same choices.
    """
    updated = deepcopy(document)
    pending = [
        revision
        for revision in updated.get("pending_revisions", [])
        if revision.get("status", "draft") == "draft"
    ]
    baseline_contacts = deepcopy(_baseline_contacts(updated))
    current_contacts = updated.get("contacts", [])
    frame_count = len(updated.get("frames", []))
    choices = updated.get("revision_choices", [])
    new_contacts = deepcopy(baseline_contacts)
    unresolved: list[dict[str, Any]] = []

    for index, contact in enumerate(new_contacts):
        covering = [revision for revision in pending if _applies_to_contact(contact, revision)]
        if not covering:
            continue
        start, end = int(contact["start"]), int(contact["end"])
        baseline_path = _baseline_path(target, updated, contact)
        missing: set[int] = set()
        for revision in covering:
            missing.update(revision_missing_frames(revision))
        current = current_contacts[index] if index < len(current_contacts) else None
        current_path = None
        if current is not None and current.get("target_path_space") == "world":
            raw_path = current.get("target_path")
            if raw_path is not None and len(raw_path) == end - start + 1:
                current_path = raw_path

        path: list[Any] = []
        frame_basis: list[tuple[int, str | None, str, Any]] = []
        for offset, anchor in enumerate(baseline_path):
            frame = start + offset
            if anchor is None or frame in missing:
                path.append(None)
                frame_basis.append((frame, None, "missing", None))
                continue
            frozen = freeze_before_frame is not None and frame < freeze_before_frame
            confirmed = confirmed_before_frame is not None and frame < confirmed_before_frame
            if frozen and current_path is not None:
                path.append(deepcopy(current_path[offset]))
                frame_basis.append((frame, None, "frozen_prefix", None))
                continue
            if frozen or confirmed:
                path.append(list(anchor))
                frame_basis.append((frame, None, "frozen_prefix", None))
                continue
            covering_frame = [
                revision
                for revision in covering
                if _revision_range(revision, frame_count)[0] <= frame <= _revision_range(revision, frame_count)[1]
            ]
            if not covering_frame:
                path.append(list(anchor))
                frame_basis.append((frame, None, "baseline", None))
                continue
            revision = None
            interpretation = None
            basis = None
            if len(covering_frame) == 1:
                revision = covering_frame[0]
                interpretation = _revision_interpretation(revision)
                basis = revision.get("basis")
                choice = _choice_for(choices, _revision_name(revision), frame)
                if choice is not None:
                    interpretation = choice["interpretation"]
                    basis = choice.get("basis", basis)
            else:
                for candidate in covering_frame:
                    choice = _choice_for(choices, _revision_name(candidate), frame)
                    if choice is not None:
                        revision = candidate
                        interpretation = choice["interpretation"]
                        basis = choice.get("basis", candidate.get("basis"))
                        break
                if revision is None:
                    unresolved.append(
                        {
                            "target": contact.get("name") or contact["joint"],
                            "frame": frame,
                            "candidates": [_revision_name(item) for item in covering_frame],
                        }
                    )
                    path.append(list(anchor))
                    frame_basis.append((frame, None, "pending_choice", None))
                    continue
            new_anchor = _transform_point(anchor, revision) if interpretation == RELATIVE else anchor
            path.append(list(new_anchor))
            frame_basis.append((frame, _revision_name(revision), interpretation, basis))

        segments: list[dict[str, Any]] = []
        for frame, revision_name, interpretation, basis in frame_basis:
            if (
                segments
                and segments[-1]["revision"] == revision_name
                and segments[-1]["interpretation"] == interpretation
                and segments[-1]["basis"] == basis
                and segments[-1]["end"] == frame - 1
            ):
                segments[-1]["end"] = frame
            else:
                segments.append(
                    {
                        "start": frame,
                        "end": frame,
                        "revision": revision_name,
                        "interpretation": interpretation,
                        "basis": basis,
                    }
                )
        jumps = []
        previous_anchor = None
        for offset, anchor in enumerate(path):
            if anchor is None:
                previous_anchor = None
                continue
            if previous_anchor is not None:
                jump = vlength(vsub(tuple(anchor), tuple(previous_anchor)))
                if jump > 0.005:
                    jumps.append(
                        {
                            "frame": start + offset,
                            "jump": jump,
                            "from": list(previous_anchor),
                            "to": list(anchor),
                        }
                    )
            previous_anchor = anchor
        contact["target_path"] = path
        contact["target_path_space"] = "world"
        contact.pop("velocity", None)
        contact["calibration"] = {
            "revisions": sorted(_revision_name(item) for item in covering),
            "segments": segments,
            "status": "draft",
            "missing_frames": sorted(frame for frame in missing if start <= frame <= end),
            "boundary_jumps": jumps,
        }

    if unresolved:
        raise ValueError(f"pending calibration conflicts require explicit choices: {unresolved}")
    updated["contacts"] = new_contacts
    updated["applied_pending_revisions"] = [_revision_name(item) for item in pending]
    return updated


def withdraw_pending_revision(document: dict[str, Any], revision_name: str, target=None) -> dict[str, Any]:
    """Withdraw a draft revision, restore baseline contacts, drop its choices."""
    updated = deepcopy(document)
    remaining = []
    removed = None
    for revision in updated.get("pending_revisions", []):
        if _revision_name(revision) == revision_name:
            if revision.get("status", "draft") != "draft":
                raise ValueError(f"published calibration cannot be withdrawn: {revision_name}")
            removed = revision
            continue
        remaining.append(revision)
    if removed is None:
        raise ValueError(f"unknown pending calibration revision: {revision_name}")
    updated["pending_revisions"] = remaining
    updated["revision_choices"] = [
        choice for choice in updated.get("revision_choices", []) if choice["revision"] != revision_name
    ]
    if "_pending_baseline_contacts" in updated:
        updated["contacts"] = deepcopy(updated["_pending_baseline_contacts"])
    updated["withdrawn_pending"] = [*updated.get("withdrawn_pending", []), revision_name]
    updated["revision_conflicts"] = detect_revision_conflicts(updated, target)
    return updated


def save_revision_state(document: dict[str, Any]) -> dict[str, Any]:
    """Snapshot pending revisions, choices and conflicts for later reopening."""
    return {
        "pending_revisions": deepcopy(document.get("pending_revisions", [])),
        "revision_choices": deepcopy(document.get("revision_choices", [])),
        "revision_conflicts": deepcopy(document.get("revision_conflicts", [])),
        "withdrawn_pending": deepcopy(document.get("withdrawn_pending", [])),
    }


def restore_revision_state(document: dict[str, Any], state: dict[str, Any], target=None) -> dict[str, Any]:
    """Reopen a saved conflict-pending state and restore every segment's basis.

    Conflicts are recomputed deterministically from the restored revisions;
    when a skeleton is supplied a mismatch with the saved conflicts is an
    error instead of silently drifting.
    """
    updated = deepcopy(document)
    updated["pending_revisions"] = deepcopy(state.get("pending_revisions", []))
    updated["revision_choices"] = deepcopy(state.get("revision_choices", []))
    updated["withdrawn_pending"] = deepcopy(state.get("withdrawn_pending", []))
    if updated["pending_revisions"]:
        updated.setdefault("_pending_baseline_contacts", deepcopy(updated.get("contacts", [])))
    recomputed = detect_revision_conflicts(updated, target)
    saved = state.get("revision_conflicts", [])
    if target is not None:
        if recomputed != saved:
            raise ValueError("saved revision conflicts do not match the recomputed conflicts")
        updated["revision_conflicts"] = recomputed
    else:
        updated["revision_conflicts"] = deepcopy(saved)
    return updated
