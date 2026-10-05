import json
import tempfile
import unittest
from pathlib import Path

from anim.contacts import apply_foot_contacts, parse_contacts
from anim.models import Motion, load_mapping, load_motion, load_skeleton
from anim.pipeline import run_batch
from anim.retarget import pose_world_positions, retarget_motion
from anim.calibration import (
    apply_calibration_revision,
    choose_calibration_interpretation,
    partial_recalculate_calibration,
    publish_calibration,
    register_calibration_revision,
    withdraw_draft_calibration,
)
from anim.schemes import compare_schemes
from anim.validation import check_contact_sliding, validate_grounded


ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "data" / "resources"


def build(skeleton_name, motion_name, mapping_name):
    source = load_skeleton(RESOURCES / "skeletons" / "simple_source.json")
    target = load_skeleton(RESOURCES / "skeletons" / f"{skeleton_name}.json")
    motion = load_motion(RESOURCES / "motions" / f"{motion_name}.json")
    mapping = load_mapping(RESOURCES / "mappings" / f"{mapping_name}.json")
    document = retarget_motion(source, target, motion, mapping).document
    return source, target, motion, mapping, document


def build_railing(skeleton_name="simple_target", mapping_name="simple_source_to_simple_target"):
    return build(skeleton_name, "railing_source", mapping_name)


def calibrated_rail_path(contact, baseline_path):
    overrides = contact.get("target_path_overrides", {})
    return [overrides.get(str(frame), overrides.get(frame, point)) for frame, point in enumerate(baseline_path, start=contact["start"])]


class OverlappingContactTests(unittest.TestCase):
    def test_overlapping_feet_and_hand_all_satisfied(self):
        _, target, _, _, document = build_railing()
        report = apply_foot_contacts(target, document, ground_scale=document["root_scale"])
        by_name = {contact["contact"]: contact for contact in report["contacts"]}
        self.assertEqual(report["unmet"], [])
        self.assertTrue(by_name["step_edge_foot"]["met"])
        self.assertTrue(by_name["step_top_foot"]["met"])
        self.assertTrue(by_name["escalator_rail"]["met"])
        # frame 2 sits inside all three intervals: overlap is really solved
        self.assertEqual(check_contact_sliding(target, Motion(document), tolerance=0.005), [])

    def test_hand_rides_moving_rail_and_foot_rides_moving_step(self):
        _, target, _, _, document = build_railing()
        report = apply_foot_contacts(target, document, ground_scale=document["root_scale"])
        moving = {c["contact"] for c in report["contacts"] if c["moving"]}
        self.assertEqual(moving, {"step_top_foot", "escalator_rail"})
        hand = next(c for c in report["contacts"] if c["contact"] == "escalator_rail")
        path = hand["anchor_path"]
        self.assertGreater(path[-1][0] - path[0][0], 0.1)
        motion = Motion(document)
        for frame_index in (3, 4):
            position = pose_world_positions(target, motion.frames[frame_index])["l_hand"]
            expected = path[frame_index - 2]
            error = sum((a - b) ** 2 for a, b in zip(position, expected)) ** 0.5
            self.assertLess(error, 0.005)

    def test_moving_platform_carries_root_and_feet(self):
        _, target, _, _, document = build("simple_target", "moving_platform_source", "simple_source_to_simple_target")
        report = apply_foot_contacts(target, document, ground_scale=document["root_scale"])
        self.assertEqual(report["unmet"], [])
        self.assertEqual(check_contact_sliding(target, Motion(document), tolerance=0.005), [])
        first_root = document["frames"][0]["root_translation"]
        last_root = document["frames"][-1]["root_translation"]
        self.assertGreater(last_root[0] - first_root[0], 0.1)
        contact = next(c for c in report["contacts"] if c["contact"] == "platform_left")
        self.assertGreater(contact["max_root_adjustment"], 0.1)

    def test_tall_target_keeps_pose_and_satisfies_contacts(self):
        _, target, _, _, document = build_railing("simple_tall_target", "simple_source_to_simple_tall_target")
        report = apply_foot_contacts(target, document, ground_scale=document["root_scale"])
        self.assertEqual(report["unmet"], [])
        self.assertEqual(check_contact_sliding(target, Motion(document), tolerance=0.005), [])
        motion = Motion(document)
        for frame in motion.frames:
            positions = pose_world_positions(target, frame)
            self.assertGreater(positions["hips"][1], positions["l_foot"][1])
            self.assertGreater(positions["hips"][1], 0.5 * document["root_scale"])


class PriorityAndBlendTests(unittest.TestCase):
    def test_unsatisfiable_contact_yields_with_error_not_success(self):
        _, target, _, _, document = build_railing()
        for contact in document["contacts"]:
            if contact.get("name") == "escalator_rail":
                contact["ground_y"] = 0.2  # rail at ankle height: unreachable
                contact["velocity"] = [0.0, 0.0, 0.0]
        report = apply_foot_contacts(target, document, ground_scale=document["root_scale"])
        by_name = {contact["contact"]: contact for contact in report["contacts"]}
        self.assertTrue(by_name["step_edge_foot"]["met"])
        self.assertTrue(by_name["step_top_foot"]["met"])
        self.assertFalse(by_name["escalator_rail"]["met"])
        self.assertGreater(by_name["escalator_rail"]["max_error"], 0.1)
        hand_unmet = [item for item in report["unmet"] if item["joint"] == "l_hand"]
        self.assertTrue(hand_unmet)
        self.assertTrue(all(item["yielded"] for item in hand_unmet))
        self.assertTrue(all(set(item["yielded_to"]) for item in hand_unmet))
        self.assertTrue(all("error" in item and "residual" in item for item in hand_unmet))

    def test_priority_order_decides_which_contact_yields(self):
        _, target, _, _, document = build(
            "simple_target", "walk_ground_source", "simple_source_to_simple_target"
        )
        # Both feet planted for the whole clip, but the left foot anchor is
        # dragged 1.5 m aside: only the higher-priority foot can be satisfied.
        scale = document["root_scale"]
        contacts = []
        for contact in document["contacts"]:
            contact["start"] = 0
            contact["end"] = 3
            contact["name"] = f"{contact['joint']}_hold"
            contacts.append(contact)
        left = next(c for c in contacts if c["joint"] == "l_foot")
        left_fk = pose_world_positions(target, document["frames"][0])["l_foot"]
        left["target_path"] = [
            [left_fk[0] + 1.5 * scale, 0.1119 * scale, left_fk[2]] for _ in range(4)
        ]
        document["contacts"] = contacts
        comparison = compare_schemes(
            target,
            document,
            [
                {"name": "right_first", "priorities": {"r_foot": 0, "l_foot": 1}},
                {"name": "left_first", "priorities": {"r_foot": 1, "l_foot": 0}},
            ],
            ground_scale=document["root_scale"],
        )
        schemes = {scheme["name"]: scheme for scheme in comparison["schemes"]}
        right_first = {c["joint"]: c for c in schemes["right_first"]["per_contact"]}
        left_first = {c["joint"]: c for c in schemes["left_first"]["per_contact"]}
        self.assertTrue(right_first["r_foot"]["met"])
        self.assertFalse(right_first["l_foot"]["met"])
        self.assertTrue(left_first["l_foot"]["met"])
        self.assertFalse(left_first["r_foot"]["met"])
        self.assertIn("l_foot", schemes["right_first"]["yielded_joints"])
        self.assertIn("r_foot", schemes["left_first"]["yielded_joints"])

    def test_blend_ramps_hand_motion_and_free_frames_stay_untouched(self):
        _, target, _, _, document = build_railing()
        original = json.loads(json.dumps(document))
        report = apply_foot_contacts(target, document, ground_scale=document["root_scale"])
        hand = next(c for c in report["contacts"] if c["contact"] == "escalator_rail")
        anchor_in = hand["anchor_path"][0]
        original_hand = pose_world_positions(target, original["frames"][2])["l_hand"]
        blended_hand = pose_world_positions(target, document["frames"][2])["l_hand"]
        # blend_in weight at the first frame is 0.5: halfway from free pose to anchor
        for component in range(3):
            midpoint = (original_hand[component] + anchor_in[component]) / 2
            self.assertAlmostEqual(blended_hand[component], midpoint, places=3)

        _, target2, _, _, walk = build("simple_target", "walk_ground_source", "simple_source_to_simple_target")
        walk["contacts"] = [
            contact for contact in walk["contacts"]
            if contact["joint"] == "r_foot"
        ]
        untouched = json.loads(json.dumps(walk))
        apply_foot_contacts(target2, walk, ground_scale=walk["root_scale"])
        for frame_index in (2, 3):
            self.assertEqual(walk["frames"][frame_index], untouched["frames"][frame_index])

    def test_invalid_moving_contact_specs_are_rejected(self):
        joints = {"l_foot"}
        with self.assertRaises(ValueError):
            parse_contacts(
                [{"joint": "l_foot", "start": 0, "end": 2, "target_path": [[0, 0, 0]]}], 4, joints
            )
        with self.assertRaises(ValueError):
            parse_contacts(
                [{
                    "joint": "l_foot",
                    "start": 0,
                    "end": 1,
                    "target_path": [[0, 0, 0], [0, 0, 0]],
                    "velocity": [1, 0, 0],
                }],
                4,
                joints,
            )


class SchemeComparisonBatchTests(unittest.TestCase):
    def test_compare_batch_saves_reports_and_recomputes_identically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "output"
            batch_path = ROOT / "data" / "batches" / "constraint_schemes_batch.json"
            first = run_batch(RESOURCES, batch_path, output)
            self.assertEqual(first["failed"], 0, first)
            reports = {}
            for name in ("turn_scheme_check", "stairs_scheme_check", "railing_scheme_check", "platform_scheme_check"):
                reports[name] = (output / "comparisons" / f"{name}.json").read_text(encoding="utf-8")
            second = run_batch(RESOURCES, batch_path, output, overwrite=True)
            self.assertEqual(second["failed"], 0, second)
            for name, content in reports.items():
                recomputed = (output / "comparisons" / f"{name}.json").read_text(encoding="utf-8")
                self.assertEqual(json.loads(recomputed), json.loads(content), name)

            railing = json.loads(reports["railing_scheme_check"])
            scheme_names = {scheme["name"] for scheme in railing["schemes"]}
            self.assertEqual(scheme_names, {"feet_first", "hand_first", "no_rail"})
            for scheme in railing["schemes"]:
                self.assertIn("per_contact", scheme)
                self.assertIn("yielded_joints", scheme)
            self.assertEqual(railing["recommended"], "feet_first")

            platform = json.loads(reports["platform_scheme_check"])
            ride = next(s for s in platform["schemes"] if s["name"] == "ride_platform")
            self.assertTrue(all(c["moving"] for c in ride["per_contact"]))
            self.assertTrue(ride["passed"])

            motion = load_motion(output / "motions" / "railing_simple_target.json")
            target = load_skeleton(RESOURCES / "skeletons" / "simple_target.json")
            self.assertEqual(check_contact_sliding(target, motion, tolerance=0.005), [])
            platform_motion = load_motion(output / "motions" / "moving_platform_target.json")
            self.assertEqual(check_contact_sliding(target, platform_motion, tolerance=0.005), [])

    def test_selected_scheme_motion_records_grounding(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "output"
            summary = run_batch(RESOURCES, ROOT / "data" / "batches" / "constraint_schemes_batch.json", output)
            job = next(item for item in summary["jobs"] if item["job"] == "railing_scheme_check")
            self.assertEqual(job["status"], "success")
            grounding = job["validation"]["grounding"]
            names = {contact["contact"] for contact in grounding["contacts"]}
            self.assertEqual(names, {"step_edge_foot", "step_top_foot", "escalator_rail"})
            self.assertTrue(all(contact["met"] for contact in grounding["contacts"]))


class ManualPinAndHandoffTests(unittest.TestCase):
    def test_manual_foot_pin_freezes_prefix_and_keeps_handoff_history(self):
        _, target, _, _, document = build(
            "simple_target", "moving_platform_source", "simple_source_to_simple_target"
        )
        pinned_position = pose_world_positions(target, document["frames"][1])["l_foot"]
        pinned_pose = json.loads(json.dumps(document["frames"][0]))
        pins = [{
            "joint": "l_foot",
            "frame": 1,
            "end": 3,
            "position": list(pinned_position),
            "reason": "user retained landing foot",
            "basis": "manual review frame 1",
        }]
        report = apply_foot_contacts(
            target,
            document,
            ground_scale=document["root_scale"],
            manual_pins=pins,
            freeze_before_frame=1,
            handoff_order=["platform_left", "platform_right"],
        )
        self.assertEqual(document["frames"][0], pinned_pose)
        for frame_index in (1, 2, 3):
            position = pose_world_positions(target, document["frames"][frame_index])["l_foot"]
            error = sum((a - b) ** 2 for a, b in zip(position, pinned_position)) ** 0.5
            self.assertLess(error, 0.02)
        self.assertEqual(document["manual_pins"][0]["basis"], "manual review frame 1")
        self.assertTrue(document["contact_history"])
        self.assertEqual(report["handoffs"]["handoff_order"], ["platform_left", "platform_right"])
        self.assertIn("timeline", report["handoffs"])
        self.assertTrue(any(row["supporters"] for row in report["handoffs"]["timeline"]))

    def test_manual_pin_reports_conflict_with_later_moving_support(self):
        _, target, _, _, document = build(
            "simple_target", "moving_platform_source", "simple_source_to_simple_target"
        )
        pinned_position = pose_world_positions(target, document["frames"][1])["l_foot"]
        pinned_position = (
            pinned_position[0] - 1.0 * document["root_scale"],
            pinned_position[1],
            pinned_position[2],
        )
        report = apply_foot_contacts(
            target,
            document,
            ground_scale=document["root_scale"],
            manual_pins=[{
                "joint": "l_foot",
                "frame": 1,
                "end": 3,
                "position": list(pinned_position),
                "basis": "manual fixed foot",
            }],
            freeze_before_frame=1,
        )
        conflicts = report["manual_pin_conflicts"]
        self.assertTrue(conflicts)
        self.assertEqual(conflicts[0]["start"], 1)
        self.assertEqual(conflicts[0]["end"], 3)
        self.assertGreater(conflicts[0]["max_pose_error"], 0.1)
        yielded = [item for item in report["unmet"] if item["joint"] == "l_foot"]
        self.assertTrue(yielded)
        self.assertIn("manual_pin:l_foot", yielded[0]["yielded_targets"])


class CalibrationRevisionTests(unittest.TestCase):
    def test_world_and_relative_recalibration_are_separate_interpretations(self):
        _, target, _, _, document = build_railing()
        revision = {
            "name": "rail_origin_v2",
            "target": "escalator_rail",
            "apply_from": 3,
            "old_origin": [0.0, 0.0, 0.0],
            "new_origin": [0.35, 0.0, 0.0],
            "rotation_y": 0.0,
            "basis": "rehearsal platform origin moved without changing contact intent",
        }
        world_pending = apply_calibration_revision(target, document, {**revision, "interpretation": "world"})
        relative_pending = apply_calibration_revision(target, document, {**revision, "interpretation": "relative"})
        self.assertEqual(world_pending["contacts"], document["contacts"])
        self.assertEqual(relative_pending["contacts"], document["contacts"])
        world_doc = choose_calibration_interpretation(target, world_pending, "rail_origin_v2", "world", basis="review")
        relative_doc = choose_calibration_interpretation(target, relative_pending, "rail_origin_v2", "relative", basis="review")
        world_rail = next(contact for contact in world_doc["contacts"] if contact.get("name") == "escalator_rail")
        relative_rail = next(contact for contact in relative_doc["contacts"] if contact.get("name") == "escalator_rail")
        baseline_path = world_doc["_calibration_contact_baseline_paths"]["escalator_rail"]
        world_path = calibrated_rail_path(world_rail, baseline_path)
        relative_path = calibrated_rail_path(relative_rail, baseline_path)
        self.assertAlmostEqual(world_path[0][0], relative_path[0][0])
        self.assertAlmostEqual(relative_path[2][0], world_path[2][0] + 0.35)
        self.assertEqual(world_rail["calibration"]["segment_basis"][1]["interpretation"], "world")
        self.assertEqual(relative_rail["calibration"]["segment_basis"][1]["interpretation"], "relative")
        self.assertTrue(all("calibration" not in contact for contact in relative_doc["contacts"] if contact.get("name") != "escalator_rail"))

    def test_revision_applies_after_confirmed_boundary_and_reports_continuity_jump(self):
        _, target, _, _, document = build_railing()
        prefix_pose = json.loads(json.dumps(document["frames"][2]))
        comparison = compare_schemes(
            target,
            document,
            [
                {
                    "name": "keep_world",
                    "calibration_revision": {
                        "name": "rail_v2_world",
                        "target": "escalator_rail",
                        "interpretation": "world",
                        "apply_from": 3,
                        "old_origin": [0, 0, 0],
                        "new_origin": [0.4, 0, 0],
                    },
                    "calibration_decisions": [{
                        "revision": "rail_v2_world",
                        "interpretation": "world",
                        "start": 3,
                    }],
                    "priorities": {"step_edge_foot": 0, "step_top_foot": 0, "escalator_rail": 1},
                },
                {
                    "name": "hand_follows_relative",
                    "calibration_revision": {
                        "name": "rail_v2_relative",
                        "target": "escalator_rail",
                        "interpretation": "relative",
                        "apply_from": 3,
                        "old_origin": [0, 0, 0],
                        "new_origin": [0.4, 0, 0],
                        "manual_choices": [{"joint": "l_foot", "choice": "keep_world_position"}],
                    },
                    "calibration_decisions": [{
                        "revision": "rail_v2_relative",
                        "interpretation": "relative",
                        "start": 3,
                    }],
                    "priorities": {"step_edge_foot": 0, "step_top_foot": 0, "escalator_rail": 1},
                    "manual_pins": [{
                        "joint": "l_foot",
                        "frame": 3,
                        "end": 5,
                        "basis": "foot remains on independently calibrated world target",
                    }],
                },
            ],
            ground_scale=document["root_scale"],
        )
        relative = next(scheme for scheme in comparison["schemes"] if scheme["name"] == "hand_follows_relative")
        self.assertEqual(relative["calibration_audit"]["apply_from"], 3)
        self.assertEqual(relative["calibration_audit"]["manual_choices"][0]["joint"], "l_foot")
        self.assertIn("baseline_metrics", relative)
        self.assertIn("calibration_delta", relative)
        rail_contact = next(contact for contact in comparison["grounding_reports"]["hand_follows_relative"]["contacts"] if contact["contact"] == "escalator_rail")
        self.assertTrue(any(jump["frame"] == 3 for jump in rail_contact["calibration"]["boundary_jumps"]))
        self.assertEqual(document["frames"][2], prefix_pose)

    def test_missing_target_record_is_not_filled_from_another_support(self):
        _, target, _, _, document = build_railing()
        revision = {
            "name": "rail_gap",
            "target": "escalator_rail",
            "interpretation": "relative",
            "apply_from": 2,
            "old_origin": [0, 0, 0],
            "new_origin": [0.25, 0, 0],
            "missing_frames": [4],
        }
        pending = apply_calibration_revision(target, document, revision)
        self.assertNotIn("target_path", next(contact for contact in pending["contacts"] if contact.get("name") == "escalator_rail"))
        revised = choose_calibration_interpretation(target, pending, "rail_gap", "relative", start=2, end=5, basis="review")
        rail = next(contact for contact in revised["contacts"] if contact.get("name") == "escalator_rail")
        revised_path = calibrated_rail_path(rail, revised["_calibration_contact_baseline_paths"]["escalator_rail"])
        self.assertIsNone(revised_path[2])
        report = apply_foot_contacts(target, revised, ground_scale=revised["root_scale"])
        gaps = report["target_recording_gaps"]
        self.assertEqual(gaps[0]["contact"], "escalator_rail")
        self.assertEqual(gaps[0]["frames"], [4])
        foot = next(contact for contact in report["contacts"] if contact["contact"] == "step_top_foot")
        self.assertTrue(foot["met"])
        self.assertEqual(check_contact_sliding(target, Motion(revised), tolerance=0.005), [])

    def test_publish_keeps_old_lineage_and_draft_withdrawal_leaves_confirmed_data(self):
        _, target, _, _, document = build_railing()
        revision = {
            "name": "rail_draft",
            "target": "escalator_rail",
            "interpretation": "relative",
            "apply_from": 2,
            "old_origin": [0, 0, 0],
            "new_origin": [0.1, 0, 0],
            "basis": "draft rehearsal measurement",
        }
        draft = apply_calibration_revision(target, document, revision)
        report = apply_foot_contacts(target, draft, ground_scale=draft["root_scale"], freeze_before_frame=2)
        confirmed_pose = json.loads(json.dumps(draft["frames"][1]))
        published = publish_calibration(draft, "rail_draft", evidence={"report": report["max_contact_error"]})
        self.assertEqual(published["calibration_publications"][0]["revision"], "rail_draft")
        self.assertEqual(published["calibration_revisions"][0]["status"], "published")
        self.assertTrue(published["calibration_publications"][0]["contact_basis"])
        withdrawn = withdraw_draft_calibration(
            apply_calibration_revision(target, document, revision), "rail_draft"
        )
        self.assertEqual(withdrawn["frames"][1], confirmed_pose)
        self.assertNotIn("rail_draft", [item.get("name") for item in withdrawn.get("calibration_revisions", [])])
        withdrawn_rail = next(contact for contact in withdrawn["contacts"] if contact.get("name") == "escalator_rail")
        self.assertNotIn("target_path", withdrawn_rail)
        self.assertIn("velocity", withdrawn_rail)
        with self.assertRaises(ValueError):
            withdraw_draft_calibration(published, "rail_draft")

    def test_overlapping_rail_revisions_remain_pending_until_explicit_choice(self):
        _, target, _, _, document = build_railing()
        span = {
            "name": "span_gap",
            "target": "escalator_rail",
            "apply_from": 3,
            "apply_until": 5,
            "old_origin": [0, 0, 0],
            "new_origin": [0.3, 0, 0],
            "missing_frames": [4],
        }
        after_gap = {
            "name": "after_gap",
            "target": "escalator_rail",
            "apply_from": 5,
            "apply_until": 5,
            "old_origin": [0, 0, 0],
            "new_origin": [0.7, 0, 0],
        }

        first_order = register_calibration_revision(target, apply_calibration_revision(target, document, span), after_gap)
        second_order = apply_calibration_revision(target, register_calibration_revision(target, document, after_gap), span)

        self.assertTrue(first_order["calibration_conflicts"])
        self.assertEqual(first_order["calibration_conflicts"][0]["segments"][0]["start"], 5)
        self.assertEqual(
            first_order["calibration_conflicts"][0]["segments"][0]["consequences"],
            second_order["calibration_conflicts"][0]["segments"][0]["consequences"],
        )
        baseline_rail = next(contact for contact in document["contacts"] if contact.get("name") == "escalator_rail")
        unresolved_rail = next(contact for contact in first_order["contacts"] if contact.get("name") == "escalator_rail")
        self.assertEqual(unresolved_rail, baseline_rail)
        self.assertNotIn("target_path", unresolved_rail)
        self.assertEqual(unresolved_rail.get("velocity"), baseline_rail.get("velocity"))

        resolved = choose_calibration_interpretation(
            target,
            first_order,
            "span_gap",
            "relative",
            start=3,
            end=4,
            basis="gap-spanning relative contact",
        )
        resolved = choose_calibration_interpretation(
            target,
            resolved,
            "after_gap",
            "relative",
            start=5,
            end=5,
            basis="reviewer selected post-gap relative calibration",
            rejected_revisions=["span_gap"],
        )
        rail = next(contact for contact in resolved["contacts"] if contact.get("name") == "escalator_rail")
        self.assertEqual(resolved["calibration_conflicts"], [])
        self.assertIsNone(rail["calibration"]["segment_basis"][0]["revision"])
        self.assertEqual(rail["calibration"]["segment_basis"][0]["end"], 2)
        self.assertEqual(rail["calibration"]["segment_basis"][1]["revision"], "span_gap")
        self.assertEqual(rail["calibration"]["segment_basis"][2]["revision"], "after_gap")
        final_path = calibrated_rail_path(rail, resolved["_calibration_contact_baseline_paths"]["escalator_rail"])
        self.assertIsNone(final_path[2])
        self.assertAlmostEqual(final_path[3][0], 0.4224389093345493 + 0.7)

    def test_pending_conflict_round_trip_and_partial_recompute_match_full_result(self):
        _, target, _, _, document = build_railing()
        span = {
            "name": "span_gap",
            "target": "escalator_rail",
            "apply_from": 3,
            "apply_until": 5,
            "old_origin": [0, 0, 0],
            "new_origin": [0.3, 0, 0],
            "missing_frames": [4],
        }
        after_gap = {
            "name": "after_gap",
            "target": "escalator_rail",
            "apply_from": 5,
            "apply_until": 5,
            "old_origin": [0, 0, 0],
            "new_origin": [0.7, 0, 0],
        }
        pending = register_calibration_revision(target, apply_calibration_revision(target, document, span), after_gap)
        reopened = json.loads(json.dumps(pending))
        self.assertEqual(reopened["calibration_conflicts"], pending["calibration_conflicts"])
        self.assertEqual(reopened["contacts"], document["contacts"])
        self.assertEqual(
            reopened["_calibration_contact_baseline"],
            document["contacts"],
        )
        partial = partial_recalculate_calibration(None, reopened)
        full = recompute_from_reopened = json.loads(json.dumps(partial))
        self.assertEqual(partial["contacts"], full["contacts"])
        self.assertEqual(partial["calibration_audit"], full["calibration_audit"])

    def test_pending_revisions_round_trip_segment_adoption_and_withdrawal(self):
        _, target, _, _, document = build_railing()
        span = {
            "name": "span_gap",
            "target": "escalator_rail",
            "apply_from": 3,
            "apply_until": 5,
            "old_origin": [0, 0, 0],
            "new_origin": [0.3, 0, 0],
            "missing_frames": [4],
        }
        after_gap = {
            "name": "after_gap",
            "target": "escalator_rail",
            "apply_from": 5,
            "apply_until": 5,
            "old_origin": [0, 0, 0],
            "new_origin": [0.7, 0, 0],
        }
        baseline_rail = next(contact for contact in document["contacts"] if contact.get("name") == "escalator_rail")
        pending = register_calibration_revision(
            target,
            apply_calibration_revision(target, document, span),
            after_gap,
        )
        with self.assertRaises(ValueError):
            register_calibration_revision(target, pending, after_gap)
        pending_rail = next(contact for contact in pending["contacts"] if contact.get("name") == "escalator_rail")
        self.assertEqual(pending_rail, baseline_rail)
        consequence = pending["calibration_conflicts"][0]["segments"][0]["consequences"]
        self.assertEqual(set(consequence["span_gap"]), {"missing", "world", "relative"})
        self.assertEqual(set(consequence["after_gap"]), {"missing", "world", "relative"})

        reopened = json.loads(json.dumps(pending))
        partial = choose_calibration_interpretation(
            target,
            reopened,
            "span_gap",
            "relative",
            start=3,
            end=4,
            basis="span adopted first",
        )
        partial_rail = next(contact for contact in partial["contacts"] if contact.get("name") == "escalator_rail")
        original_path = partial["_calibration_contact_baseline_paths"]["escalator_rail"]
        partial_path = calibrated_rail_path(partial_rail, original_path)
        self.assertNotIn("target_path", partial_rail)
        self.assertEqual(partial_rail.get("velocity"), baseline_rail.get("velocity"))
        self.assertEqual(partial_path[0], original_path[0])
        self.assertIsNone(partial_path[2])
        self.assertEqual(partial_path[3], original_path[3])
        self.assertAlmostEqual(partial_path[1][0], original_path[1][0] + 0.3)

        adopted = choose_calibration_interpretation(
            target,
            partial,
            "after_gap",
            "relative",
            start=5,
            end=5,
            basis="post-gap adopted second",
            rejected_revisions=["span_gap"],
        )
        adopted_rail = next(contact for contact in adopted["contacts"] if contact.get("name") == "escalator_rail")
        adopted_path = calibrated_rail_path(adopted_rail, original_path)
        self.assertEqual(adopted["calibration_conflicts"], [])
        self.assertAlmostEqual(adopted_path[3][0], original_path[3][0] + 0.7)
        with self.assertRaises(ValueError):
            choose_calibration_interpretation(
                target,
                adopted,
                "span_gap",
                "relative",
                start=3,
                end=5,
                basis="cannot reclaim confirmed post-gap interval",
            )

        without_after = withdraw_draft_calibration(adopted, "after_gap")
        remaining_rail = next(contact for contact in without_after["contacts"] if contact.get("name") == "escalator_rail")
        remaining_path = calibrated_rail_path(remaining_rail, original_path)
        self.assertEqual(remaining_path[3], original_path[3])
        self.assertAlmostEqual(remaining_path[1][0], original_path[1][0] + 0.3)
        restored = withdraw_draft_calibration(without_after, "span_gap")
        restored_rail = next(contact for contact in restored["contacts"] if contact.get("name") == "escalator_rail")
        self.assertEqual(restored_rail, baseline_rail)

    def test_withdrawal_order_restores_remaining_revisions_from_original_basis(self):
        _, target, _, _, document = build_railing()
        span = {
            "name": "span_gap",
            "target": "escalator_rail",
            "apply_from": 3,
            "apply_until": 5,
            "old_origin": [0, 0, 0],
            "new_origin": [0.3, 0, 0],
            "missing_frames": [4],
        }
        after_gap = {
            "name": "after_gap",
            "target": "escalator_rail",
            "apply_from": 5,
            "apply_until": 5,
            "old_origin": [0, 0, 0],
            "new_origin": [0.7, 0, 0],
        }
        first = register_calibration_revision(target, apply_calibration_revision(target, document, span), after_gap)
        second = apply_calibration_revision(target, register_calibration_revision(target, document, after_gap), span)
        withdrawn_first_then_second = withdraw_draft_calibration(withdraw_draft_calibration(first, "span_gap"), "after_gap")
        withdrawn_second_then_first = withdraw_draft_calibration(withdraw_draft_calibration(second, "after_gap"), "span_gap")
        baseline_rail = next(contact for contact in document["contacts"] if contact.get("name") == "escalator_rail")
        for result in (withdrawn_first_then_second, withdrawn_second_then_first):
            rail = next(contact for contact in result["contacts"] if contact.get("name") == "escalator_rail")
            self.assertNotIn("target_path", rail)
            self.assertEqual(rail.get("velocity"), baseline_rail.get("velocity"))

    def test_split_confirmation_of_same_revision_keeps_earlier_decisions(self):
        _, target, _, _, document = build_railing()
        revision = {
            "name": "rail_piecewise",
            "target": "escalator_rail",
            "apply_from": 2,
            "old_origin": [0, 0, 0],
            "new_origin": [0.3, 0, 0],
            "basis": "rail origin surveyed in two passes",
        }
        pending = apply_calibration_revision(target, document, revision)
        first = choose_calibration_interpretation(
            target, pending, "rail_piecewise", "relative", start=2, end=2, basis="frame 2 keeps relative intent"
        )
        second = choose_calibration_interpretation(
            target, first, "rail_piecewise", "world", start=4, end=4, basis="frame 4 keeps world anchor"
        )
        self.assertEqual(len(second["calibration_decisions"]), 2)
        kept = {(d["start"], d["end"], d["interpretation"]) for d in second["calibration_decisions"]}
        self.assertEqual(kept, {(2, 2, "relative"), (4, 4, "world")})
        rail = next(contact for contact in second["contacts"] if contact.get("name") == "escalator_rail")
        baseline = second["_calibration_contact_baseline_paths"]["escalator_rail"]
        path = calibrated_rail_path(rail, baseline)
        self.assertAlmostEqual(path[0][0], baseline[0][0] + 0.3)
        self.assertEqual(path[1], baseline[1])
        self.assertEqual(path[2], baseline[2])
        segment_basis = rail["calibration"]["segment_basis"]
        self.assertEqual(segment_basis[0]["interpretation"], "relative")
        self.assertEqual(segment_basis[0]["start"], 2)
        self.assertIsNone(segment_basis[1]["revision"])
        self.assertEqual(segment_basis[2]["interpretation"], "world")
        self.assertEqual(segment_basis[2]["start"], 4)
        repeated = choose_calibration_interpretation(
            target, second, "rail_piecewise", "world", start=4, end=4, basis="frame 4 keeps world anchor"
        )
        self.assertEqual(len(repeated["calibration_decisions"]), 2)
        repeated_rail = next(contact for contact in repeated["contacts"] if contact.get("name") == "escalator_rail")
        self.assertEqual(repeated_rail, rail)
        with self.assertRaises(ValueError):
            choose_calibration_interpretation(
                target, second, "rail_piecewise", "world", start=4, end=5, basis="overlaps confirmed frame 4"
            )

    def test_cross_gap_split_confirmation_survives_reopen_and_withdrawal(self):
        _, target, _, _, document = build_railing()
        span = {
            "name": "span_gap",
            "target": "escalator_rail",
            "apply_from": 3,
            "apply_until": 5,
            "old_origin": [0, 0, 0],
            "new_origin": [0.3, 0, 0],
            "missing_frames": [4],
        }
        after_gap = {
            "name": "after_gap",
            "target": "escalator_rail",
            "apply_from": 5,
            "apply_until": 5,
            "old_origin": [0, 0, 0],
            "new_origin": [0.7, 0, 0],
        }
        pending = register_calibration_revision(
            target,
            apply_calibration_revision(target, document, span),
            after_gap,
        )
        first = choose_calibration_interpretation(
            target, pending, "span_gap", "relative", start=3, end=4, basis="pre-gap segment relative"
        )
        reopened = json.loads(json.dumps(first))
        second = choose_calibration_interpretation(
            target, reopened, "span_gap", "world", start=5, end=5,
            basis="post-gap frame keeps world anchor",
            rejected_revisions=["after_gap"],
        )
        self.assertEqual(len(second["calibration_decisions"]), 2)
        rail = next(contact for contact in second["contacts"] if contact.get("name") == "escalator_rail")
        baseline = second["_calibration_contact_baseline_paths"]["escalator_rail"]
        path = calibrated_rail_path(rail, baseline)
        self.assertAlmostEqual(path[1][0], baseline[1][0] + 0.3)
        self.assertIsNone(path[2])
        self.assertEqual(path[3], baseline[3])
        self.assertEqual(second["calibration_conflicts"], [])
        without_after = withdraw_draft_calibration(second, "after_gap")
        remaining_rail = next(contact for contact in without_after["contacts"] if contact.get("name") == "escalator_rail")
        remaining_path = calibrated_rail_path(remaining_rail, baseline)
        self.assertEqual(len(without_after["calibration_decisions"]), 2)
        self.assertAlmostEqual(remaining_path[1][0], baseline[1][0] + 0.3)
        self.assertIsNone(remaining_path[2])
        self.assertEqual(remaining_path[3], baseline[3])


if __name__ == "__main__":
    unittest.main()
