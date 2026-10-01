from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import load_mapping, load_motion, load_skeleton
from .plans import compare_contact_plans, save_plan_document
from .pipeline import run_batch
from .playback import play_pose_samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="anim-pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-batch", help="Run a retargeting batch")
    run.add_argument("--resources", required=True)
    run.add_argument("--batch", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--overwrite", action="store_true")
    preview = sub.add_parser("preview", help="Evaluate target motion with forward kinematics")
    preview.add_argument("--skeleton", required=True)
    preview.add_argument("--motion", required=True)
    compare = sub.add_parser("compare-plans", help="Compare contact constraint plans")
    compare.add_argument("--resources", required=True)
    compare.add_argument("--source-skeleton", required=True)
    compare.add_argument("--target-skeleton", required=True)
    compare.add_argument("--source-motion", required=True)
    compare.add_argument("--mapping", required=True)
    compare.add_argument("--plans", required=True, help="JSON file with a plans list")
    compare.add_argument("--save-selected")
    compare.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "run-batch":
        summary = run_batch(args.resources, args.batch, args.output, overwrite=args.overwrite)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0 if summary["failed"] == 0 else 1
    if args.command == "preview":
        skeleton = load_skeleton(args.skeleton)
        motion = load_motion(args.motion)
        if motion.skeleton != skeleton.name:
            raise SystemExit("motion does not reference the supplied skeleton")
        print(json.dumps(play_pose_samples(skeleton, motion), indent=2, ensure_ascii=False))
        return 0
    if args.command == "compare-plans":
        base = Path(args.resources)
        source = load_skeleton(base / args.source_skeleton)
        target = load_skeleton(base / args.target_skeleton)
        motion = load_motion(base / args.source_motion)
        mapping = load_mapping(base / args.mapping)
        plans_document = json.loads(Path(args.plans).read_text(encoding="utf-8"))
        plans = plans_document.get("plans", plans_document)
        result = compare_contact_plans(source, target, motion, mapping, plans)
        if args.save_selected:
            chosen = next(
                evaluation
                for evaluation in result["evaluations"]
                if evaluation["name"] == result["selected"]
            )
            save_plan_document(chosen, args.save_selected, overwrite=args.overwrite)
        printable = {key: value for key, value in result.items() if key != "evaluations"}
        print(json.dumps(printable, indent=2, ensure_ascii=False))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
