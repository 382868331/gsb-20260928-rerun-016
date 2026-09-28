import json
import tempfile
import unittest
from pathlib import Path

from anim.models import Motion, load_mapping, load_motion, load_skeleton
from anim.pipeline import run_batch
from anim.retarget import mapped_skeleton_scale, pose_world_positions, retarget_motion
from anim.validation import validate_retarget


ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "data" / "resources"


class RetargetingTests(unittest.TestCase):
    def setUp(self):
        self.source = load_skeleton(RESOURCES / "skeletons" / "simple_source.json")
        self.target = load_skeleton(RESOURCES / "skeletons" / "simple_target.json")
        self.motion = load_motion(RESOURCES / "motions" / "walk_source.json")
        self.mapping = load_mapping(RESOURCES / "mappings" / "simple_source_to_simple_target.json")

    def test_playable_target_motion_passes_physical_checks(self):
        result = retarget_motion(self.source, self.target, self.motion, self.mapping)
        target_motion = Motion(result.document)
        report = validate_retarget(self.source, self.target, self.motion, target_motion, self.mapping)
        self.assertTrue(report["passed"], report)

    def test_root_translation_and_bone_length_scale_together(self):
        result = retarget_motion(self.source, self.target, self.motion, self.mapping)
        scale = result.document["root_scale"]
        expected_scale = mapped_skeleton_scale(
            self.source, self.target, self.mapping.required_pairs()
        )
        self.assertAlmostEqual(scale, expected_scale)
        self.assertGreater(scale, 1.24)
        self.assertLess(scale, 1.26)
        self.assertEqual(result.document["frames"][-1]["root_translation"][2], 0.45 * scale)

    def test_forward_kinematics_emits_target_joint_positions(self):
        result = retarget_motion(self.source, self.target, self.motion, self.mapping)
        target_motion = type(self.motion)(result.document)
        positions = pose_world_positions(self.target, target_motion.frames[0])
        self.assertIn("l_foot", positions)
        scale = result.document["root_scale"]
        self.assertAlmostEqual(positions["hips"][1], 1.0 * scale, places=6)
        self.assertLess(positions["l_foot"][1], positions["hips"][1])
        self.assertIn("back_gear", result.document["retargeting"]["target_only_joints"])
        self.assertIn("head_top", result.document["retargeting"]["missing_source_joints"])

    def test_failed_batch_does_not_overwrite_existing_motion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output = temp_path / "output"
            first = run_batch(RESOURCES, ROOT / "data" / "batches" / "demo_batch.json", output)
            self.assertEqual(first["failed"], 0)
            converted = output / "motions" / "walk_simple_target.json"
            original_bytes = converted.read_bytes()

            strict_target = json.loads((RESOURCES / "skeletons" / "simple_target.json").read_text())
            strict_target["joints"]["l_knee"]["rotation_limits"]["x"] = [0.0, 0.0]
            strict_path = temp_path / "strict_target.json"
            strict_path.write_text(json.dumps(strict_target), encoding="utf-8")
            batch = {
                "kind": "retarget_batch",
                "name": "failing_batch",
                "jobs": [{
                    "name": "must_fail",
                    "source_skeleton": "skeletons/simple_source.json",
                    "target_skeleton": str(strict_path),
                    "source_motion": "motions/walk_source.json",
                    "mapping": "mappings/simple_source_to_simple_target.json",
                    "output_motion": "motions/walk_simple_target.json",
                }],
            }
            batch_path = temp_path / "failing_batch.json"
            batch_path.write_text(json.dumps(batch), encoding="utf-8")
            failed = run_batch(RESOURCES, batch_path, output)
            self.assertEqual(failed["failed"], 1)
            self.assertEqual(converted.read_bytes(), original_bytes)


if __name__ == "__main__":
    unittest.main()
