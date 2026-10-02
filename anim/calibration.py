from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .math3d import (
    Vec,
    q_rotate_vector,
    quat_from_euler_xyz,
    vadd,
    vlength,
    vsub,
)


INTERPRETATIONS = ("world", "relative")
STATUSES = ("draft", "published")


@dataclass(frozen=True)
class CalibrationRevision:
    """A recalibration proposal for already established contact targets.

    ``world`` keeps every anchor at its old world position (the contact
    intention is interpreted as a fixed world spot); ``relative`` moves the
    anchors with the support's new origin/orientation, i.e. the contact keeps
    its position *relative to the recalibrated moving support*.
    The two interpretations are kept side by side instead of blindly moving
    every contact point to new coordinates.
    """

    revision_id: str
    applies_from: int
    applies_until: int | None
    interpretation: str
    origin_offset: Vec = (0.0, 0.0, 0.0)
    rotation_euler: Vec = (0.0, 0.0, 0.0)
    contacts: tuple[str, ...] | None = None
    status: str = "draft"
    version: str | None = None
    basis: str | None = None
    reason: str | None = None

    def active(self, frame_index: int) -> bool:
        if frame_index < self.applies_from:
            return False
        return self.applies_until is None or frame_index <= self.applies_until

    def transform_point(self, point: Vec, ground_scale: float = 1.0) -> Vec:
        if self.interpretation == "world":
            return point
        offset = tuple(component * ground_scale for component in self.origin_offset)
        rotation = quat_from_euler_xyz(self.rotation_euler)
        return vadd(q_rotate_vector(rotation, point), offset)

    def transform_velocity(self, velocity: Vec) -> Vec:
        if self.interpretation == "world":
            return (0.0, 0.0, 0.0)
        rotation = quat_from_euler_xyz(self.rotation_euler)
        return q_rotate_vector(rotation, velocity)


def parse_calibration_revisions(
    raw_revisions: list[dict[str, Any]] | None,
    frame_count: int,
) -> list[CalibrationRevision]:
    revisions: list[CalibrationRevision] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_revisions or []):
        revision_id = raw.get("id") or raw.get("revision_id")
        if not revision_id:
            raise ValueError(f"calibration revision {index} requires an id")
        if revision_id in seen:
            raise ValueError(f"duplicate calibration revision id: {revision_id}")
        seen.add(revision_id)
        interpretation = raw.get("interpretation", "relative")
        if interpretation not in INTERPRETATIONS:
            raise ValueError(
                f"calibration revision {revision_id} interpretation must be one of {INTERPRETATIONS}"
            )
        status = raw.get("status", "draft")
        if status not in STATUSES:
            raise ValueError(f"calibration revision {revision_id} status must be one of {STATUSES}")
        applies_from = int(raw.get("applies_from", 0))
        applies_until = raw.get("applies_until")
        applies_until = int(applies_until) if applies_until is not None else None
        if applies_from < 0 or applies_from >= frame_count:
            raise ValueError(f"calibration revision {revision_id} applies_from outside the clip")
        if applies_until is not None and (applies_until < applies_from or applies_until >= frame_count):
            raise ValueError(f"calibration revision {revision_id} has an invalid applies_until")
        offset = raw.get("origin_offset", [0.0, 0.0, 0.0])
        rotation = raw.get("rotation_euler", raw.get("rotation", [0.0, 0.0, 0.0]))
        if len(offset) != 3 or len(rotation) != 3:
            raise ValueError(f"calibration revision {revision_id} offset/rotation must have 3 components")
        contacts = raw.get("contacts")
        if contacts is not None:
            contacts = tuple(str(key) for key in contacts)
        target = raw.get("target")
        if contacts is None and target is not None:
            contacts = (str(target),)
        if status == "published" and not raw.get("version"):
            raise ValueError(f"published calibration revision {revision_id} requires a version")
        revisions.append(
            CalibrationRevision(
                revision_id=str(revision_id),
                applies_from=applies_from,
                applies_until=applies_until,
                interpretation=interpretation,
                origin_offset=tuple(float(value) for value in offset),
                rotation_euler=tuple(float(value) for value in rotation),
                contacts=contacts,
                status=status,
                version=raw.get("version"),
                basis=raw.get("basis"),
                reason=raw.get("reason"),
            )
        )
    return revisions


def revision_matches(revision: CalibrationRevision, contact_key: str, joint: str) -> bool:
    if revision.contacts is None:
        return True
    return contact_key in revision.contacts or joint in revision.contacts


def active_revision(
    revisions: list[CalibrationRevision],
    contact_key: str,
    joint: str,
    frame_index: int,
) -> CalibrationRevision | None:
    """Return the newest matching revision active on a frame.

    A contact that starts before ``applies_from`` keeps the original
    calibration on its early frames; when several revisions match, the one
    introduced latest wins, mirroring an amendment history.
    """
    candidates = [
        revision
        for revision in revisions
        if revision.active(frame_index) and revision_matches(revision, contact_key, joint)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda revision: revision.applies_from)


def revision_boundaries(
    revisions: list[CalibrationRevision],
    contact_key: str,
    joint: str,
    interval_start: int,
    interval_end: int,
) -> list[int]:
    """Frames where an interval spanning a calibration cut changes version."""
    boundaries: set[int] = set()
    matching = [r for r in revisions if revision_matches(r, contact_key, joint)]
    for revision in matching:
        frame = revision.applies_from
        if interval_start < frame <= interval_end:
            boundaries.add(frame)
        if revision.applies_until is not None and interval_start <= revision.applies_until < interval_end:
            boundaries.add(revision.applies_until + 1)
    return sorted(boundaries)


def recheck_transfer_context(
    document: dict[str, Any],
    revisions: list[CalibrationRevision],
    contact_anchor_paths: dict[str, list[Vec | None]],
    *,
    ground_scale: float,
    slide_tolerance: float = 0.005,
) -> dict[str, Any]:
    """Re-verify time scaling, skeleton proportion and loop seams under revisions.

    Anchor trajectories are authored per frame, so a time-scaled source must
    still supply one anchor per contact frame; skeleton rescaling is checked
    by the scale applied to origin offsets; cyclic contacts get a seam jump
    check between the last and first anchor after recalibration.
    """
    frame_count = len(document["frames"])
    frame_rate = float(document.get("frame_rate", 30.0))
    anchor_checks: list[dict[str, Any]] = []
    seam_violations: list[dict[str, Any]] = []
    for raw_contact in document.get("contacts", []):
        key = raw_contact.get("name") or raw_contact["joint"]
        path = contact_anchor_paths.get(key, [])
        expected_length = int(raw_contact["end"]) - int(raw_contact["start"]) + 1
        gap_frames = [
            int(raw_contact["start"]) + offset
            for offset, point in enumerate(path)
            if point is None
        ]
        anchor_checks.append({
            "contact": key,
            "anchor_frames": len(path),
            "expected_frames": expected_length,
            "gap_frames": gap_frames,
            "complete": len(path) == expected_length and not gap_frames,
        })
        if raw_contact.get("loop") and len(path) >= 2 and path[0] is not None and path[-1] is not None:
            seam_gap = vlength(vsub(tuple(path[-1]), tuple(path[0])))
            if seam_gap > slide_tolerance:
                seam_violations.append({
                    "contact": key,
                    "seam_gap": seam_gap,
                    "tolerance": slide_tolerance,
                })
    return {
        "frame_rate": frame_rate,
        "clip_frames": frame_count,
        "source_duration_seconds": frame_count / max(frame_rate, 1e-9),
        "skeleton_root_scale": ground_scale,
        "mapping_version": document.get("mapping_version"),
        "anchor_coverage": anchor_checks,
        "loop_seam_violations": seam_violations,
        "revision_ids": [revision.revision_id for revision in revisions],
    }
