#!/usr/bin/env python3
"""Fit per-regime logistic regression (α, λ, b) on labeled operator scores.

Input JSONL rows need:
    regime, label (0/1), operator_scores, features: {beta, nu, kappa}

This is the paper's Stage-3 training: scikit-learn LogisticRegression with
L2, C=0.1, random_state=42, fitted independently per regime.

    python scripts/fit_calibration.py --input outputs/train_scored.jsonl \
        --output configs/fitted_weights.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prove.calibration import Calibration, PAPER_ALPHA, PAPER_INTERCEPT
from prove.features import paired_feature
from prove.regime import REGIME_OPERATORS, REGIME_NAMES


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def design_matrix(rows: list[dict], operators: list[str], phi_bar: dict[str, float]):
    X, y = [], []
    for row in rows:
        scores = row.get("operator_scores") or {}
        feats = row.get("features") or {}
        if any(op not in scores for op in operators):
            continue
        label = row.get("label", row.get("hallu_label"))
        if label is None:
            continue
        beta, nu, kappa = float(feats["beta"]), float(feats["nu"]), float(feats["kappa"])
        main = [float(scores[op]) for op in operators]
        inter = []
        for op in operators:
            phi = paired_feature(op, beta, nu, kappa)
            inter.append((phi - phi_bar[op]) * float(scores[op]))
        X.append(main + inter)
        y.append(int(label))
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def phi_means(rows: list[dict], operators: list[str]) -> dict[str, float]:
    acc = defaultdict(list)
    for row in rows:
        feats = row.get("features") or {}
        if "beta" not in feats:
            continue
        beta, nu, kappa = float(feats["beta"]), float(feats["nu"]), float(feats["kappa"])
        for op in operators:
            acc[op].append(paired_feature(op, beta, nu, kappa))
    return {op: float(np.mean(vals)) if vals else 0.0 for op, vals in acc.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit PROVE calibration")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--C", type=float, default=0.1, help="L2 inverse regularization (paper: 0.1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_rows(args.input)
    by_regime: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        regime = str(row.get("regime", "")).lower()
        if regime in REGIME_NAMES:
            by_regime[regime].append(row)

    calib = Calibration(
        alpha={r: dict(w) for r, w in PAPER_ALPHA.items()},
        intercept=dict(PAPER_INTERCEPT),
    )

    for regime in REGIME_NAMES:
        ops = list(REGIME_OPERATORS[regime])
        subset = by_regime.get(regime, [])
        print(f"{regime}: {len(subset)} rows, ops={ops}")
        if len(subset) < 10:
            print("  skip (not enough rows); keeping paper defaults")
            continue
        means = phi_means(subset, ops)
        X, y = design_matrix(subset, ops, means)
        if len(y) < 10 or y.sum() == 0 or y.sum() == len(y):
            print("  skip (degenerate labels); keeping paper defaults")
            continue
        clf = LogisticRegression(C=args.C, max_iter=1000, random_state=args.seed)
        clf.fit(X, y)
        coef = clf.coef_[0]
        alpha = {op: float(coef[i]) for i, op in enumerate(ops)}
        lambdas = {op: float(coef[len(ops) + i]) for i, op in enumerate(ops)}
        intercept = float(clf.intercept_[0])
        calib.alpha[regime] = alpha
        calib.lambdas[regime] = lambdas
        calib.phi_bar[regime] = means
        calib.intercept[regime] = intercept
        print(f"  alpha={ {k: round(v, 3) for k, v in alpha.items()} }")
        print(f"  lambda={ {k: round(v, 3) for k, v in lambdas.items()} }")
        print(f"  intercept={intercept:.3f}")
        print(f"  phi_bar={ {k: round(v, 3) for k, v in means.items()} }")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = calib.to_dict()
    payload["C"] = args.C
    payload["seed"] = args.seed
    payload["n_train"] = {r: len(by_regime[r]) for r in REGIME_NAMES}
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
