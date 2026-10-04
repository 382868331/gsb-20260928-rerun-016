import json
import unittest
from pathlib import Path

from anim.contacts import apply_foot_contacts
from anim.models import load_mapping, load_motion, load_skeleton
from anim.retarget import retarget_motion
from anim.revisions import (
    apply_pending_revisions,
    choose_segment_interpretation,
    detect_revision_conflicts,
    register_pending_revision,
    restore_revision_state,
    save_revision_state,
    withdraw_pending_revision,
)


ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "data" / "resources"


def build_railing():
    source = load_skeleton(RESOURCES / "skeletons" / "simple_source.json")
    target = load_skeleton(RESOURCES / "skeletons" / "simple_target.json")
    motion = load_motion(RESOURCES / "motions" / "railing_source.json")
    mapping = load_mapping(RESOURCES / "mappings" / "simple_source_to_simple_target.json")
    document = retarget_motion(source, target, motion, mapping).document
    return target, document


def rail(document):
    return next(contact for contact in document["contacts"] if contact.get("name") == "escalator_rail")


def revision_world():
    return {
        "name": "rail_world",
        "target": "escalator_rail",
        "interpretation": "world",
        "apply_from": 3,
        "old_origin": [0.0, 0.0, 0.0],
        "new_origin": [0.4, 0.0, 0.0],
        "basis": "survey says rail origin is already in world space",
    }


def revision_relative():
    return {
        "name": "rail_relative",
        "target": "escalator_rail",
        "interpretation": "relative",
        "apply_from": 3,
        "old_origin": [0.0, 0.0, 0.0],
        "new_origin": [0.4, 0.0, 0.0],
        "basis": "rehearsal platform shifted 0.4 m, keep contact intent",
    }


class PendingRevisionConflictTests(unittest.TestCase):
    def test_overlapping_revisions_wait_for_explicit_choice(self):
        target, document = build_railing()
        original_contacts = json.loads(json.dumps(document["contacts"]))
        pending = register_pending_revision(document, revision_world(), target)
        pending = register_pending_revision(pending, revision_relative(), target)

        conflicts = pending["revision_conflicts"]
        self.assertEqual(len(conflicts), 1)
        conflict = conflicts[0]
        self.assertEqual(conflict["status"], "pending_choice")
        self.assertEqual((conflict["start"], conflict["end"]), (3, 5))
        self.assertEqual({option["revision"] for option in conflict["options"]}, {"rail_world", "rail_relative"})
        by_name = {option["revision"]: option for option in conflict["options"]}
        self.assertEqual(by_name["rail_world"]["consequence"]["anchor_shift"], 0.0)
        self.assertAlmostEqual(by_name["rail_relative"]["consequence"]["anchor_shift"], 0.4)
        self.assertTrue(all(option["basis"] for option in conflict["options"]))
        # registration alone never rewrites contacts
        self.assertEqual(pending["contacts"], original_contacts)

        with self.assertRaises(ValueError):
            apply_pending_revisions(target, pending)

        chosen = choose_segment_interpretation(pending, "rail_relative", 3, 5, "relative")
        applied = apply_pending_revisions(target, chosen)
        rail_contact = rail(applied)
        self.assertNotIn("target_path", original_contacts[2])  # untouched source of truth
        # world-keep prefix, relative-shifted suffix on the moving rail
        world_applied = apply_pending_revisions(
            target, choose_segment_interpretation(pending, "rail_world", 3, 5, "world")
        )
        world_path = rail(world_applied)["target_path"]
        relative_path = rail_contact["target_path"]
        self.assertAlmostEqual(relative_path[0][0], world_path[0][0], places=9)
        for offset in (1, 2, 3):
            self.assertAlmostEqual(relative_path[offset][0] - world_path[offset][0], 0.4, places=9)
        segments = rail_contact["calibration"]["segments"]
        chosen_segment = next(s for s in segments if s["revision"] == "rail_relative")
        self.assertEqual((chosen_segment["start"], chosen_segment["end"]), (3, 5))
        self.assertEqual(chosen_segment["interpretation"], "relative")
        self.assertEqual(chosen_segment["basis"], "rehearsal platform shifted 0.4 m, keep contact intent")
        # other contacts stay traceable and untouched
        feet = [c for c in applied["contacts"] if c.get("name") != "escalator_rail"]
        self.assertTrue(all("calibration" not in contact for contact in feet))
        self.assertEqual(feet, [c for c in original_contacts if c.get("name") != "escalator_rail"])

    def test_registration_order_does_not_decide_outcome(self):
        target, document = build_railing()
        first = register_pending_revision(document, revision_world(), target)
        first = register_pending_revision(first, revision_relative(), target)
        second = register_pending_revision(document, revision_relative(), target)
        second = register_pending_revision(second, revision_world(), target)
        self.assertEqual(
            {c["start"] for c in first["revision_conflicts"]},
            {c["start"] for c in second["revision_conflicts"]},
        )
        for doc in (first, second):
            chosen = choose_segment_interpretation(doc, "rail_world", 3, 5, "world")
            applied = apply_pending_revisions(target, chosen)
            self.assertEqual(rail(applied)["target_path"], rail(apply_pending_revisions(
                target, choose_segment_interpretation(first, "rail_world", 3, 5, "world")
            ))["target_path"])
        chosen_first = apply_pending_revisions(
            target, choose_segment_interpretation(first, "rail_world", 3, 5, "world")
        )
        chosen_second = apply_pending_revisions(
            target, choose_segment_interpretation(second, "rail_world", 3, 5, "world")
        )
        self.assertEqual(chosen_first["contacts"], chosen_second["contacts"])


class ObservationGapTests(unittest.TestCase):
    def gap_spanning_revision(self):
        return {
            "name": "rail_span_gap",
            "target": "escalator_rail",
            "interpretation": "relative",
            "apply_from": 2,
            "old_origin": [0.0, 0.0, 0.0],
            "new_origin": [0.25, 0.0, 0.0],
            "missing_frames": [4],
            "basis": "rail camera lost frame 4",
        }

    def post_gap_revision(self):
        return {
            "name": "rail_post_gap",
            "target": "escalator_rail",
            "interpretation": "relative",
            "apply_from": 5,
            "old_origin": [0.0, 0.0, 0.0],
            "new_origin": [0.25, 0.0, 0.0],
            "missing_frames": [4],
            "basis": "post-gap re-survey",
        }

    def test_gap_is_never_filled_and_orders_agree(self):
        target, document = build_railing()
        spanning = register_pending_revision(document, self.gap_spanning_revision(), target)
        both = register_pending_revision(spanning, self.post_gap_revision(), target)
        # the spanning revision and the post-gap revision overlap at frame 5:
        # the conflict is listed and waits for an explicit choice
        self.assertEqual(
            [(c["start"], c["end"]) for c in both["revision_conflicts"]], [(5, 5)]
        )
        with self.assertRaises(ValueError):
            apply_pending_revisions(target, both)
        chosen = choose_segment_interpretation(both, "rail_post_gap", 5, 5, "relative")
        applied_both = apply_pending_revisions(target, chosen)
        applied_spanning = apply_pending_revisions(target, spanning)
        path_both = rail(applied_both)["target_path"]
        path_spanning = rail(applied_spanning)["target_path"]
        # frame 4 (offset 2) stays an observation gap in both
        self.assertIsNone(path_both[2])
        self.assertIsNone(path_spanning[2])
        # post-gap frame agrees between gap-spanning and post-gap revisions
        self.assertEqual(path_both[3], path_spanning[3])
        self.assertEqual(rail(applied_both)["calibration"]["missing_frames"], [4])
        # solving must not invent contact for the gap either
        report = apply_foot_contacts(target, applied_both, ground_scale=applied_both["root_scale"])
        gaps = report["target_recording_gaps"]
        self.assertEqual(gaps[0]["contact"], "escalator_rail")
        self.assertEqual(gaps[0]["frames"], [4])
        # consequences for the contested frame are shown side by side
        options = both["revision_conflicts"][0]["options"]
        self.assertEqual({o["revision"] for o in options}, {"rail_span_gap", "rail_post_gap"})
        self.assertTrue(all(o["consequence"]["anchor_shift"] is not None for o in options))

    def test_adoption_and_withdrawal_orders_are_equivalent(self):
        target, document = build_railing()
        # order 1: adopt both, withdraw the post-gap one
        both = register_pending_revision(document, self.gap_spanning_revision(), target)
        both = register_pending_revision(both, self.post_gap_revision(), target)
        both = choose_segment_interpretation(both, "rail_post_gap", 5, 5, "relative")
        applied = apply_pending_revisions(target, both)
        withdrawn = withdraw_pending_revision(applied, "rail_post_gap", target)
        reapplied = apply_pending_revisions(target, withdrawn)
        # order 2: only ever adopt the gap-spanning revision
        spanning_only = register_pending_revision(document, self.gap_spanning_revision(), target)
        applied_spanning = apply_pending_revisions(target, spanning_only)
        self.assertEqual(rail(reapplied)["target_path"], rail(applied_spanning)["target_path"])
        self.assertEqual(withdrawn["withdrawn_pending"], ["rail_post_gap"])

        # reverse withdrawal: adopt both, withdraw the spanning one
        withdrawn2 = withdraw_pending_revision(applied, "rail_span_gap", target)
        reapplied2 = apply_pending_revisions(target, withdrawn2)
        post_only = register_pending_revision(document, self.post_gap_revision(), target)
        applied_post = apply_pending_revisions(target, post_only)
        self.assertEqual(rail(reapplied2)["target_path"], rail(applied_post)["target_path"])
        # post-gap-only revision leaves pre-gap frames at baseline and gap empty
        path = rail(applied_post)["target_path"]
        self.assertIsNone(path[2])
        self.assertEqual(rail(applied_post)["calibration"]["segments"][0]["interpretation"], "baseline")


class SaveReopenTests(unittest.TestCase):
    def test_conflict_pending_state_survives_save_and_reopen(self):
        target, document = build_railing()
        pending = register_pending_revision(document, revision_world(), target)
        pending = register_pending_revision(pending, revision_relative(), target)
        pending = choose_segment_interpretation(pending, "rail_relative", 3, 4, "relative")
        state = json.loads(json.dumps(save_revision_state(pending)))

        _, fresh = build_railing()
        reopened = restore_revision_state(fresh, state, target)
        self.assertEqual(reopened["revision_conflicts"], pending["revision_conflicts"])
        self.assertEqual(reopened["revision_choices"], pending["revision_choices"])
        self.assertEqual(
            detect_revision_conflicts(reopened, target), pending["revision_conflicts"]
        )
        # per-segment basis is restored and still drives application
        reopened = choose_segment_interpretation(reopened, "rail_relative", 5, 5, "world")
        applied = apply_pending_revisions(target, reopened)
        segments = rail(applied)["calibration"]["segments"]
        relative_segment = next(s for s in segments if s["interpretation"] == "relative")
        self.assertEqual((relative_segment["start"], relative_segment["end"]), (3, 4))
        self.assertEqual(relative_segment["basis"], "rehearsal platform shifted 0.4 m, keep contact intent")
        world_segment = next(s for s in segments if s["interpretation"] == "world")
        self.assertEqual((world_segment["start"], world_segment["end"]), (5, 5))

        # tampered conflict state is rejected instead of silently drifting
        tampered = json.loads(json.dumps(state))
        tampered["revision_conflicts"][0]["end"] += 1
        with self.assertRaises(ValueError):
            restore_revision_state(fresh, tampered, target)


class LocalRecomputeTests(unittest.TestCase):
    def test_local_recompute_matches_full_compute(self):
        target, document = build_railing()
        pending = register_pending_revision(document, revision_relative(), target)
        full = apply_pending_revisions(target, pending)
        # local recompute: freeze the confirmed prefix, re-derive the rest
        local = apply_pending_revisions(target, full, freeze_before_frame=4)
        self.assertEqual(rail(local)["target_path"], rail(full)["target_path"])
        segments = rail(local)["calibration"]["segments"]
        frozen = next(s for s in segments if s["interpretation"] == "frozen_prefix")
        self.assertEqual((frozen["start"], frozen["end"]), (2, 3))
        relative = next(s for s in segments if s["interpretation"] == "relative")
        self.assertEqual((relative["start"], relative["end"]), (4, 5))
        # frozen prefix really keeps the previously solved anchors
        prefix_full = rail(full)["target_path"][:2]
        prefix_local = rail(local)["target_path"][:2]
        self.assertEqual(prefix_full, prefix_local)


if __name__ == "__main__":
    unittest.main()
