from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import load_motion, load_skeleton
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
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
