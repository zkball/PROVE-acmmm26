"""Deterministic tests for proof features, routing, and published risk math."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prove.calibration import Calibration, sigmoid
from prove.features import compute_proof_features, estimate_answer_set_size
from prove.lexicon import lexicon_size
from prove.regime import REGIME_OPERATORS, operators_for_regime


class FeatureTests(unittest.TestCase):
    def test_lexicon_is_about_300_terms(self):
        n = lexicon_size()
        self.assertGreaterEqual(n, 250)
        self.assertLessEqual(n, 400)

    def test_yesno_cardinality(self):
        self.assertEqual(
            estimate_answer_set_size("Is there a fracture in the pelvis?", "No"),
            2,
        )

    def test_modality_cardinality(self):
        self.assertEqual(
            estimate_answer_set_size("What modality is used to take this image?", "CT"),
            6,
        )

    def test_beta_modality_ct_matches_paper_example(self):
        feats = compute_proof_features("What modality is used to take this image?", "CT")
        # 0.5 * (1/log2(7) + 1/2) ≈ 0.428
        expected = 0.5 * (1.0 / math.log2(7) + 0.5)
        self.assertAlmostEqual(feats["beta"], expected, places=4)

    def test_nu_fracture_pelvis(self):
        feats = compute_proof_features("Is there a fracture in the pelvis?", "No")
        # fracture + pelvis → 2 matches → 1 - 1/3 = 2/3
        self.assertAlmostEqual(feats["nu"], 2.0 / 3.0, places=4)
        self.assertEqual(feats["n_visual_terms"], 2)

    def test_kappa_atomic_answer(self):
        feats = compute_proof_features("Is there a fracture in the pelvis?", "No")
        self.assertEqual(feats["n_clauses"], 0)
        self.assertEqual(feats["kappa"], 0.0)

    def test_kappa_multi_clause(self):
        feats = compute_proof_features(
            "Describe the findings.",
            "There is pleural effusion, consolidation, and cardiomegaly",
        )
        self.assertGreaterEqual(feats["n_clauses"], 2)
        self.assertGreater(feats["kappa"], 0.5)


class RoutingTests(unittest.TestCase):
    def test_table1_routing(self):
        self.assertEqual(
            operators_for_regime("perceptual"),
            ("consistency", "contrast", "grounding"),
        )
        self.assertEqual(
            operators_for_regime("attribute"),
            ("contrast", "grounding", "sufficiency"),
        )
        self.assertEqual(
            operators_for_regime("compositional"),
            ("contrast", "grounding", "sufficiency", "claim_decomp"),
        )
        self.assertEqual(set(REGIME_OPERATORS), {"perceptual", "attribute", "compositional"})


class CalibrationTests(unittest.TestCase):
    def test_sigmoid(self):
        self.assertAlmostEqual(sigmoid(0.0), 0.5, places=6)
        self.assertGreater(sigmoid(2.0), 0.8)
        self.assertLess(sigmoid(-2.0), 0.2)

    def test_paper_perceptual_all_zero_ops(self):
        calib = Calibration.paper_default()
        out = calib.risk(
            "perceptual",
            {"consistency": 0.0, "contrast": 0.0, "grounding": 0.0},
            {"beta": 0.4, "nu": 0.3, "kappa": 0.0},
        )
        self.assertAlmostEqual(out["logit"], -2.29, places=4)
        self.assertAlmostEqual(out["risk"], sigmoid(-2.29), places=5)

    def test_paper_perceptual_consistency_alarm(self):
        calib = Calibration.paper_default()
        out = calib.risk(
            "perceptual",
            {"consistency": 1.0, "contrast": 0.0, "grounding": 0.0},
            {"beta": 0.4, "nu": 0.3, "kappa": 0.0},
        )
        self.assertAlmostEqual(out["logit"], 1.43 - 2.29, places=4)


if __name__ == "__main__":
    unittest.main()
