from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .calibration import parse_calibration_revisions, recheck_transfer_context
from .contacts import apply_foot_contacts
from .models import Motion, Skeleton, atomic_write_json
from .retarget import retarget_motion
from .validation import validate_grounded


def build_published_calibration(
    *,
    source_skeleton: Any,
    target: Skeleton,
    source_motion: Any,
    mapping: Any,
    scheme: dict[str, Any],
) -> dict[str, Any]:
    """Resolve a scheme from scratch and assemble reproducible evidence.

    Every published calibration keeps its own manual selections (pins,
    priorities, handoff order) and its own revision list; the bundle records
    source motion, mapping version and the re-checked time-scale / skeleton
    proportion / loop seam context, so an old published calibration remains
    reproducible after newer ones ship.
    """
    retargeted = retarget_motion(source_skeleton, target, source_motion, mapping)
    document = retargeted.document
    ground_scale = document["root_scale"]
    grounding = apply_foot_contacts(
        target,
        document,
        ground_scale=ground_scale,
        manual_pins=scheme.get("manual_pins"),
        handoff_order=scheme.get("handoff_order"),
        freeze_before_frame=scheme.get("freeze_before_frame"),
        calibration_revisions=scheme.get("calibration_revisions"),
    )
    validation = validate_grounded(target, Motion(document), grounding, slide_tolerance=0.005)
    anchor_paths = {
        raw_contact.get("name") or raw_contact["joint"]: [
            None if point is None else tuple(point)
            for point in raw_contact.get("anchor_path", [])
        ]
        for raw_contact in document.get("contacts", [])
    }
    revisions = parse_calibration_revisions(
        scheme.get("calibration_revisions"), len(document["frames"])
    )
    recheck = recheck_transfer_context(
        document, revisions, anchor_paths, ground_scale=ground_scale
    )
    return {
        "document": document,
        "grounding": grounding,
        "validation": validation,
        "recheck": recheck,
    }


def publish_calibration_bundle(
    output_dir: str | Path,
    job_name: str,
    scheme: dict[str, Any],
    evidence: dict[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write an immutable calibration release plus the motion it produced.

    Releases are content-addressed by their declared version and never
    overwrite one another: withdrawing a later unpublished draft cannot
    change an already released calibration, and an old release can still be
    reproduced on its own.
    """
    revision = scheme.get("calibration_revisions", [{}])[0]
    version = revision.get("version") or scheme["name"]
    bundle_dir = Path(output_dir) / "calibrations" / job_name / version
    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "kind": "published_calibration",
        "job": job_name,
        "scheme_name": scheme["name"],
        "version": version,
        "published_at": created_at,
        "source_motion": evidence["document"].get("source_motion"),
        "mapping_version": evidence["document"].get("mapping_version"),
        "target_skeleton": evidence["document"].get("target_skeleton"),
        "root_scale": evidence["document"].get("root_scale"),
        "scheme": scheme,
        "recheck": evidence["recheck"],
        "validation_passed": evidence["validation"]["passed"],
        "validation": evidence["validation"],
    }
    atomic_write_json(bundle_dir / "manifest.json", manifest, overwrite=overwrite)
    atomic_write_json(
        bundle_dir / "contact_evidence.json",
        {
            "grounding": evidence["grounding"],
            "calibration_boundaries": evidence["grounding"].get("calibration_boundaries", []),
            "calibration_conflict_summary": {
                "gap_frames": {
                    contact["contact"]: contact.get("target_record_gaps", [])
                    for contact in evidence["grounding"].get("contacts", [])
                },
            },
        },
        overwrite=overwrite,
    )
    motion_path = bundle_dir / "motion.json"
    atomic_write_json(motion_path, evidence["document"], overwrite=overwrite)
    return {
        "bundle": str(bundle_dir),
        "version": version,
        "motion": str(motion_path),
        "manifest": manifest,
    }
