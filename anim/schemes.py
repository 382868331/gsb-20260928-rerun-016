from __future__ import annotations

from copy import deepcopy
from typing import Any

from .contacts import apply_foot_contacts
from .handoffs import attach_handoff_audit
from .models import Motion, Skeleton
from .validation import validate_grounded


def apply_scheme_to_contacts(
    contacts: list[dict[str, Any]], scheme: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return contact definitions adjusted by a named constraint scheme.

    A scheme may reprioritise joints, override blend ramps and disable
    contacts entirely, so alternative trade-offs can be compared without
    editing the source resource.
    """
    disabled = set(scheme.get("disable", []))
    priorities = scheme.get("priorities", {})
    adjusted: list[dict[str, Any]] = []
    for contact in contacts:
        if contact["joint"] in disabled or contact.get("name") in disabled:
            continue
        updated = dict(contact)
        key = contact.get("name") if contact.get("name") in priorities else contact["joint"]
        if key in priorities:
            updated["priority"] = int(priorities[key])
        for field in ("blend_in", "blend_out"):
            if field in scheme:
                updated[field] = int(scheme[field])
        adjusted.append(updated)
    return adjusted


def evaluate_scheme(
    target: Skeleton,
    motion_data: dict[str, Any],
    scheme: dict[str, Any],
    *,
    ground_scale: float = 1.0,
    slide_tolerance: float = 0.005,
) -> dict[str, Any]:
    """Run one constraint scheme on a copy of the motion and collect metrics."""
    if "name" not in scheme:
        raise ValueError("constraint scheme requires a name")
    document = deepcopy(motion_data)
    document["contacts"] = apply_scheme_to_contacts(document.get("contacts", []), scheme)
    grounding = apply_foot_contacts(
        target,
        document,
        ground_scale=ground_scale,
        manual_pins=document.get("manual_pins"),
        freeze_before=(
            max((int(pin["freeze_before"]) for pin in document.get("manual_pins", [])), default=None)
        ),
    )
    validation = validate_grounded(target, Motion(document), grounding, slide_tolerance=slide_tolerance)
    attach_handoff_audit(target, document, grounding)
    yielded = sorted({item["joint"] for item in grounding["unmet"] if item.get("yielded")})
    return {
        "name": scheme["name"],
        "passed": validation["passed"],
        "contact_count": len(grounding["contacts"]),
        "max_contact_error": grounding["max_contact_error"],
        "unmet_frames": len(grounding["unmet"]),
        "yielded_joints": yielded,
        "slide_violations": validation["metrics"]["contact_slide_violations"],
        "max_root_adjustment": max(
            (contact["max_root_adjustment"] for contact in grounding["contacts"]),
            default=0.0,
        ),
        "per_contact": [
            {
                "joint": contact["joint"],
                "contact": contact.get("contact"),
                "start": contact["start"],
                "end": contact["end"],
                "priority": contact["priority"],
                "moving": contact["moving"],
                "max_error": contact["max_error"],
                "met": contact["met"],
            }
            for contact in grounding["contacts"]
        ],
        "document": document,
        "grounding": grounding,
        "validation": validation,
    }


def compare_schemes(
    target: Skeleton,
    motion_data: dict[str, Any],
    schemes: list[dict[str, Any]],
    *,
    ground_scale: float = 1.0,
    slide_tolerance: float = 0.005,
) -> dict[str, Any]:
    """Evaluate alternative constraint schemes and rank them.

    Every scheme is solved from the same unmodified retargeted motion, so the
    saved comparison can be recomputed deterministically to re-check turning,
    stair and moving-support scenarios after data changes.
    """
    if not schemes:
        raise ValueError("compare requires at least one constraint scheme")
    names = [scheme.get("name") for scheme in schemes]
    if len(set(names)) != len(names):
        raise ValueError("constraint scheme names must be unique")
    evaluations = [
        evaluate_scheme(
            target, motion_data, scheme, ground_scale=ground_scale, slide_tolerance=slide_tolerance
        )
        for scheme in schemes
    ]

    def rank(evaluation: dict[str, Any]) -> tuple[float, int, float, int, float]:
        return (
            0.0 if evaluation["passed"] else 1.0,
            evaluation["unmet_frames"],
            evaluation["max_contact_error"],
            -evaluation["contact_count"],
            evaluation["max_root_adjustment"],
        )

    recommended = min(evaluations, key=rank)["name"]
    return {
        "kind": "constraint_scheme_comparison",
        "motion": motion_data.get("name"),
        "target_skeleton": target.name,
        "ground_scale": ground_scale,
        "slide_tolerance": slide_tolerance,
        "recommended": recommended,
        "handoff_audits": {
            evaluation["name"]: evaluation["document"].get("contact_handoffs")
            for evaluation in evaluations
        },
        "schemes": [
            {key: value for key, value in evaluation.items() if key not in {"document", "grounding", "validation"}}
            for evaluation in evaluations
        ],
        "documents": {evaluation["name"]: evaluation["document"] for evaluation in evaluations},
        "grounding_reports": {evaluation["name"]: evaluation["grounding"] for evaluation in evaluations},
    }
