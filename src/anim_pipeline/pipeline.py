from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .io import atomic_write_json, load_animation, load_mapping, load_skeleton
from .retarget import retarget
from .validate import validate_animation


@dataclass
class JobResult:
    output: str | None
    passed: bool
    errors: list[str]


def run_batch(root: str | Path, batch: dict) -> list[JobResult]:
    root = Path(root)
    results: list[JobResult] = []
    for job in batch["jobs"]:
        output_path = root / job["output_animation"]
        if output_path.exists():
            results.append(
                JobResult(
                    str(output_path),
                    False,
                    [f"refusing to overwrite existing resource: {output_path}"],
                )
            )
            continue
        try:
            source_skeleton = load_skeleton(root / job["source_skeleton"])
            target_skeleton = load_skeleton(root / job["target_skeleton"])
            mapping = load_mapping(root / job["mapping"])
            source_animation = load_animation(root / job["source_animation"])
            result = retarget(source_skeleton, target_skeleton, source_animation, mapping)
            report = validate_animation(
                target_skeleton,
                result.animation,
                source_animation=source_animation,
                source_skeleton=source_skeleton,
                scale=result.scale,
            )
            if not report.passed:
                results.append(JobResult(None, False, report.errors))
                continue
            payload = result.animation.to_dict()
            payload["metadata"]["validation"] = report.to_dict()
            payload["metadata"]["unmapped_source_joints"] = result.unmapped_source
            payload["metadata"]["unmapped_target_joints"] = result.unmapped_target
            atomic_write_json(output_path, payload)
            results.append(JobResult(str(output_path), True, []))
        except Exception as exc:
            results.append(JobResult(None, False, [f"{type(exc).__name__}: {exc}"]))
    return results
