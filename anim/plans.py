from __future__ import annotations

from typing import Any

from .contacts import apply_contact_constraints
from .models import Mapping, Motion, Skeleton, atomic_write_json
from .retarget import retarget_motion
from .validation import validate_grounded, validate_retarget


def evaluate_contact_plan(
    source: Skeleton,
    target: Skeleton,
    source_motion: Motion,
    mapping: Mapping,
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Recompute retargeting + grounding for one contact constraint plan.

    The plan carries a name and a contacts list; every entry may set a
    priority, entry/exit blend windows and per-frame target keyframes for
    moving supports.  Nothing is written by this call so plans can be
    compared side by side.
    """
    retargeted = retarget_motion(source, target, source_motion, mapping)
    document = retargeted.document
    retarget_validation = validate_retarget(source, target, source_motion, Motion(document), mapping)
    if not retarget_validation["passed"]:
        raise ValueError(
            f"retargeting validation failed: {retarget_validation['failure_count']} issue(s)"
        )

    contacts = plan.get("contacts", [])
    plan_name = plan.get("name", "plan")
    grounding = apply_contact_constraints(
        target,
        document,
        ground_scale=document["root_scale"],
        contact_plan=contacts,
        plan_name=plan_name,
    )
    grounded_validation = validate_grounded(
        target, Motion(document), grounding, slide_tolerance=0.005
    )
    intervals = grounding["contacts"]
    return {
        "name": plan_name,
        "document": document,
        "retarget_validation": retarget_validation,
        "grounding_report": grounding,
        "validation": grounded_validation,
        "metrics": {
            "frames": len(document["frames"]),
            "root_scale": document["root_scale"],
            "contact_count": len(intervals),
            "satisfied": sum(1 for item in intervals if item["status"] == "satisfied"),
            "conceded": sum(1 for item in intervals if item["status"] == "conceded"),
            "unmet": sum(1 for item in intervals if item["status"] == "unmet"),
            "unmet_frames": len(grounding["unmet"]),
            "concession_frames": len(grounding["concessions"]),
            "max_contact_error": grounding["max_contact_error"],
            "slide_violations": grounded_validation["failure_count"],
        },
    }


def compare_contact_plans(
    source: Skeleton,
    target: Skeleton,
    source_motion: Motion,
    mapping: Mapping,
    plans: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate every plan from a fresh retarget and rank them for selection."""
    evaluations = [
        evaluate_contact_plan(source, target, source_motion, mapping, plan) for plan in plans
    ]

    def rank_key(item: dict[str, Any]) -> tuple[Any, ...]:
        metrics = item["metrics"]
        return (
            metrics["slide_violations"],
            metrics["unmet"],
            metrics["conceded"],
            metrics["max_contact_error"],
            metrics["unmet_frames"],
            metrics["concession_frames"],
        )

    order = sorted(range(len(evaluations)), key=lambda index: rank_key(evaluations[index]))
    best_index = order[0]
    comparisons = []
    for index, item in enumerate(evaluations):
        comparisons.append({
            "name": item["name"],
            "rank": order.index(index) + 1,
            "selected": index == best_index,
            "metrics": item["metrics"],
            "intervals": [
                {
                    "joint": interval["joint"],
                    "status": interval["status"],
                    "priority": interval["priority"],
                    "max_error": interval["max_error"],
                    "conceded_to": interval["conceded_to"],
                    "unmet_frames": interval["unmet_frames"],
                    "moving": interval["moving"],
                }
                for interval in item["grounding_report"]["contacts"]
            ],
            "unmet": item["grounding_report"]["unmet"],
            "concessions": item["grounding_report"]["concessions"],
        })
    return {
        "source_motion": source_motion.name,
        "target_skeleton": target.name,
        "evaluations": evaluations,
        "comparison": comparisons,
        "selected": evaluations[best_index]["name"],
    }


def save_plan_document(result: dict[str, Any], output_path: str, *, overwrite: bool = False) -> str:
    """Persist the chosen plan's motion and return the written path."""
    atomic_write_json(output_path, result["document"], overwrite=overwrite)
    return str(output_path)
