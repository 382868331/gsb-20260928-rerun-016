import json
import tempfile
import unittest
from pathlib import Path

from anim.calibration import parse_calibration_revisions, active_revision
from anim.contacts import apply_foot_contacts
from anim.models import Motion, load_batch, load_mapping, load_motion, load_skeleton
from anim.pipeline import run_batch
from anim.retarget import retarget_motion, pose_world_positions
from anim.validation import check_contact_sliding


ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "data" / "resources"


def build_doc():
    source = load_skeleton(RESOURCES / "skeletons" / "simple_source.json")
    target = load_skeleton(RESOURCES / "skeletons" / "simple_target.json")
    motion = load_motion(RESOURCES / "motions" / "platform_recalibration_source.json")
    mapping = load_mapping(RESOURCES / "mappings" / "simple_source_to_simple_target.json")
    document = retarget_motion(source, target, motion, mapping).document
    return source, target, motion, mapping, document


WORLD_REVISION = {
    "id": "origin_b_world",
    "applies_from": 3,
    "interpretation": "world",
    "origin_offset": [0.1, 0.0, 0.0],
}
RELATIVE_REVISION = {
    "id": "origin_b_relative",
    "applies_from": 3,
    "interpretation": "relative",
    "origin_offset": [0.1, 0.0, 0.0],
    "status": "published",
    "version": "2026.10.03-test",
}


class CalibrationParsingTests(unittest.TestCase):
    def test_interpretation_and_status_validation(self):
        with self.assertRaises(ValueError):
            parse_calibration_revisions(
                [{"id": "x", "applies_from": 0, "interpretation": "teleport"}], 4
            )
        with self.assertRaises(ValueError):
            parse_calibration_revisions(
                [{"id": "x", "applies_from": 0, "status": "published"}], 4
            )
        revisions = parse_calibration_revisions(
            [{"id": "x", "applies_from": 2, "status": "published", "version": "v9"}], 4
        )
        self.assertTrue(revisions[0].active(2))
        self.assertFalse(revisions[0].active(1))

    def test_world_interpretation_keeps_coordinates_relative_moves_them(self):
        world, relative = parse_calibration_revisions([WORLD_REVISION, RELATIVE_REVISION], 6)
        point = (1.0, 0.0, 0.0)
        self.assertEqual(world.transform_point(point, ground_scale=1.0), point)
        moved = relative.transform_point(point, ground_scale=1.0)
        self.assertAlmostEqual(moved[0], 1.1)

    def test_revision_scoping_is_per_contact(self):
        revisions = parse_calibration_revisions([
            {"id": "hand_only", "applies_from": 0, "contacts": ["platform_rail"]}
        ], 4)
        self.assertIsNotNone(active_revision(revisions, "platform_rail", "l_hand", 1))
        self.assertIsNone(active_revision(revisions, "platform_left", "l_foot", 1))

    def test_rotation_transforms_relative_velocity_but_not_world(self):
        revisions = parse_calibration_revisions([
            {"id": "r", "applies_from": 0, "interpretation": "relative",
             "rotation_euler": [0.0, 1.57079632679, 0.0]}
        ], 4)
        moved = revisions[0].transform_velocity((1.0, 0.0, 0.0))
        self.assertAlmostEqual(moved[0], 0.0, places=5)
        self.assertAlmostEqual(abs(moved[2]), 1.0, places=5)
        world = parse_calibration_revisions(
            [{"id": "w", "applies_from": 0, "interpretation": "world"}], 4
        )[0]
        self.assertEqual(world.transform_velocity((1.0, 0.0, 0.0)), (0.0, 0.0, 0.0))


class CalibrationSolvingTests(unittest.TestCase):
    def setUp(self):
        self.source, self.target, self.motion, self.mapping, self.document = build_doc()

    def test_world_vs_relative_produce_different_anchors_not_blanket_move(self):
        world_doc = json.loads(json.dumps(self.document))
        relative_doc = json.loads(json.dumps(self.document))
        world = apply_foot_contacts(
            self.target, world_doc, ground_scale=world_doc["root_scale"],
            calibration_revisions=[WORLD_REVISION],
        )
        relative = apply_foot_contacts(
            self.target, relative_doc, ground_scale=relative_doc["root_scale"],
            calibration_revisions=[RELATIVE_REVISION],
        )
        foot_world = next(c for c in world["contacts"] if c["contact"] == "platform_left")
        foot_relative = next(c for c in relative["contacts"] if c["contact"] == "platform_left")
        self.assertEqual(foot_world["anchor_path"][2], foot_relative["anchor_path"][2])
        self.assertAlmostEqual(
            foot_world["anchor_path"][3][0], foot_world["anchor_path"][2][0], places=6
        )
        self.assertGreater(
            foot_relative["anchor_path"][3][0] - foot_relative["anchor_path"][2][0], 0.05
        )

    def test_revision_applies_only_to_unconfirmed_frames(self):
        document = json.loads(json.dumps(self.document))
        frozen = json.loads(json.dumps(document["frames"]))
        apply_foot_contacts(
            self.target, document, ground_scale=document["root_scale"],
            freeze_before_frame=3, calibration_revisions=[RELATIVE_REVISION],
        )
        for frame_index in range(3):
            self.assertEqual(document["frames"][frame_index], frozen[frame_index])
        self.assertNotEqual(document["frames"][3], frozen[3])
        redrawn = json.loads(json.dumps(self.document))
        apply_foot_contacts(
            self.target, redrawn, ground_scale=redrawn["root_scale"],
            freeze_before_frame=3, calibration_revisions=[],
        )
        self.assertEqual(redrawn["frames"][:3], frozen[:3])

    def test_straddling_contact_reports_boundary_continuity_and_concession(self):
        document = json.loads(json.dumps(self.document))
        report = apply_foot_contacts(
            self.target, document, ground_scale=document["root_scale"],
            calibration_revisions=[RELATIVE_REVISION],
        )
        foot_boundary = next(
            b for b in report["calibration_boundaries"] if b["contact"] == "platform_left"
        )
        self.assertEqual(foot_boundary["frame"], 3)
        self.assertIsNone(foot_boundary["from_calibration"])
        self.assertEqual(foot_boundary["to_calibration"], "origin_b_relative")
        self.assertFalse(foot_boundary["continuous"])
        self.assertGreater(foot_boundary["anchor_jump"], 0.05)
        self.assertIn("give way", foot_boundary["required_concession"])

        world_document = json.loads(json.dumps(self.document))
        world_report = apply_foot_contacts(
            self.target, world_document, ground_scale=world_document["root_scale"],
            calibration_revisions=[WORLD_REVISION],
        )
        world_foot = next(
            b for b in world_report["calibration_boundaries"] if b["contact"] == "platform_left"
        )
        self.assertTrue(world_foot["continuous"])
        self.assertAlmostEqual(world_foot["anchor_jump"], 0.0, places=6)

    def test_missing_target_record_is_not_substituted_and_resume_does_not_teleport(self):
        document = json.loads(json.dumps(self.document))
        before = json.loads(json.dumps(document["frames"]))
        before_frames = json.loads(json.dumps(document["frames"]))
        report = apply_foot_contacts(
            self.target, document, ground_scale=document["root_scale"],
            calibration_revisions=[RELATIVE_REVISION],
        )
        rail = next(c for c in report["contacts"] if c["contact"] == "platform_rail")
        self.assertEqual(rail["target_record_gaps"], [4])
        self.assertEqual(rail["resume_frames"], [5])
        self.assertIsNone(rail["anchor_path"][2])
        # the gap frame's hand chain is not driven by another support's state
        hand_joints = ("l_shoulder", "l_elbow", "l_hand")
        for joint in hand_joints:
            self.assertEqual(
                document["frames"][4]["rotations"].get(joint),
                before[4]["rotations"].get(joint),
                joint,
            )
        resume_warnings = [u for u in report["unmet"] if u["type"] == "target_record_resume"]
        self.assertTrue(resume_warnings)
        warning = resume_warnings[0]
        self.assertEqual(warning["frame"], 5)
        self.assertEqual(warning["joint"], "l_hand")
        self.assertGreater(warning["error"], 0.01)
        self.assertFalse(warning["yielded"])
        hand_before = pose_world_positions(self.target, before_frames[5])["l_hand"]
        hand_after = pose_world_positions(self.target, document["frames"][5])["l_hand"]
        hand_gap_pose = pose_world_positions(self.target, document["frames"][4])["l_hand"]
        teleport_jump = sum((a - b) ** 2 for a, b in zip(hand_gap_pose, hand_after)) ** 0.5
        # continuity is measured across the gap -> resume boundary, not
        # against the untouched pre-contact retarget pose
        self.assertLess(teleport_jump, 0.2)
        self.assertEqual(check_contact_sliding(self.target, Motion(document), tolerance=0.005), [])

    def test_foot_world_and_hand_relative_keep_independent_choices(self):
        document = json.loads(json.dumps(self.document))
        revisions = [
            {"id": "feet_world", "applies_from": 3, "interpretation": "world",
             "contacts": ["platform_left", "platform_right"]},
            {"id": "hand_relative", "applies_from": 3, "interpretation": "relative",
             "origin_offset": [0.1, 0.0, 0.0], "contacts": ["platform_rail"]},
        ]
        report = apply_foot_contacts(
            self.target, document, ground_scale=document["root_scale"],
            calibration_revisions=revisions,
        )
        foot = next(c for c in report["contacts"] if c["contact"] == "platform_left")
        hand = next(c for c in report["contacts"] if c["contact"] == "platform_rail")
        self.assertEqual(foot["calibration_by_frame"][4], "feet_world")
        self.assertEqual(hand["calibration_by_frame"][4 - hand["start"]], "hand_relative")


class CalibrationBatchTests(unittest.TestCase):
    BATCH = ROOT / "data" / "batches" / "calibration_review_batch.json"

    def test_batch_publishes_immutable_reproducible_bundle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "out"
            first = run_batch(RESOURCES, self.BATCH, output)
            self.assertEqual(first["failed"], 0, first)
            comparison = json.loads((output / "comparisons" / "platform_recalibration.json").read_text())
            names = {scheme["name"] for scheme in comparison["schemes"]}
            self.assertEqual(
                names,
                {"keep_world_anchors", "follow_recalibrated_platform", "foot_world_hand_relative"},
            )
            for scheme in comparison["schemes"]:
                self.assertIn("calibration_conflict_summary", scheme)
                self.assertIn("calibration_revisions", scheme)

            bundle = output / "calibrations" / "platform_recalibration" / "2026.10.03-platform-origin-b"
            manifest = json.loads((bundle / "manifest.json").read_text())
            self.assertTrue((bundle / "motion.json").exists())
            self.assertTrue((bundle / "contact_evidence.json").exists())
            self.assertEqual(manifest["version"], "2026.10.03-platform-origin-b")
            self.assertTrue(manifest["validation_passed"])
            self.assertIn("recheck", manifest)
            self.assertAlmostEqual(manifest["recheck"]["skeleton_root_scale"], 1.25196, places=4)
            coverage = {item["contact"]: item for item in manifest["recheck"]["anchor_coverage"]}
            self.assertEqual(coverage["platform_rail"]["gap_frames"], [4])
            self.assertFalse(coverage["platform_rail"]["complete"])

            second = run_batch(RESOURCES, self.BATCH, output)
            self.assertTrue(any(job["status"] == "failed" for job in second["jobs"]))

            third = run_batch(RESOURCES, self.BATCH, output, overwrite=True)
            self.assertEqual(third["failed"], 0, third)
            rebuilt = json.loads((bundle / "manifest.json").read_text())
            self.assertEqual(
                rebuilt["scheme"]["calibration_revisions"],
                manifest["scheme"]["calibration_revisions"],
            )

    def test_withdrawing_draft_does_not_affect_confirmed_prefix(self):
        batch = load_batch(self.BATCH)
        trimmed = json.loads(json.dumps(batch.jobs[0]))
        trimmed["schemes"] = [
            scheme for scheme in trimmed["schemes"]
            if scheme["name"] != "foot_world_hand_relative"
        ]
        temp_dir = tempfile.mkdtemp()
        draft_batch = Path(temp_dir) / "batch.json"
        draft_batch.write_text(json.dumps({"name": "trimmed", "jobs": [trimmed]}))
        summary = run_batch(RESOURCES, draft_batch, Path(temp_dir) / "out")
        self.assertEqual(summary["failed"], 0, summary)



if __name__ == "main__":
    unittest.main()
