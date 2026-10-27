"""Stage 3: adaptive risk calibration (paper Eqs. 4–6).

    w_k(q) = α_k^r + λ_k^r · (φ_k(q,a) − φ̄_k^r)
    s(q)   = σ( Σ_k w_k(q) · op_k(q) + b_r )

Default α and b are the published VQA-RAD-fitted weights (main paper Table 2).
Default λ = 0, which recovers regime routing without the per-question delta
(paper ablation: 0.816 AUROC). Fit λ with scripts/fit_calibration.py.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .features import paired_feature

# Published base weights from the main paper (Table 2), fitted on VQA-RAD.
PAPER_ALPHA: dict[str, dict[str, float]] = {
    "perceptual": {
        "consistency": 1.43,
        "contrast": 1.40,
        "grounding": 1.00,
    },
    "attribute": {
        "contrast": 0.78,
        "grounding": 1.29,
        "sufficiency": 0.23,
    },
    "compositional": {
        "contrast": 0.37,
        "grounding": 0.83,
        "sufficiency": 0.57,
        "claim_decomp": 0.62,
    },
}

PAPER_INTERCEPT: dict[str, float] = {
    "perceptual": -2.29,
    "attribute": -1.67,
    "compositional": -1.26,
}


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class Calibration:
    """Per-regime α, λ, φ̄, and intercept."""

    alpha: dict[str, dict[str, float]] = field(
        default_factory=lambda: {r: dict(w) for r, w in PAPER_ALPHA.items()}
    )
    lambdas: dict[str, dict[str, float]] = field(default_factory=dict)
    phi_bar: dict[str, dict[str, float]] = field(default_factory=dict)
    intercept: dict[str, float] = field(default_factory=lambda: dict(PAPER_INTERCEPT))

    def __post_init__(self) -> None:
        for regime, weights in self.alpha.items():
            self.lambdas.setdefault(regime, {op: 0.0 for op in weights})
            self.phi_bar.setdefault(regime, {op: 0.0 for op in weights})
            for op in weights:
                self.lambdas[regime].setdefault(op, 0.0)
                self.phi_bar[regime].setdefault(op, 0.0)

    @classmethod
    def paper_default(cls) -> "Calibration":
        return cls()

    @classmethod
    def from_json(cls, path: str | Path) -> "Calibration":
        data = json.loads(Path(path).read_text())
        alpha: dict[str, dict[str, float]] = {}
        lambdas: dict[str, dict[str, float]] = {}
        phi_bar: dict[str, dict[str, float]] = {}
        intercept: dict[str, float] = {}
        regimes = data.get("regimes", data)
        for regime, cfg in regimes.items():
            if not isinstance(cfg, dict):
                continue
            if "alpha" in cfg:
                alpha[regime] = {k: float(v) for k, v in cfg["alpha"].items()}
                lambdas[regime] = {k: float(v) for k, v in cfg.get("lambda", {}).items()}
                phi_bar[regime] = {k: float(v) for k, v in cfg.get("phi_bar", {}).items()}
                intercept[regime] = float(cfg.get("intercept", 0.0))
        if not alpha:
            raise ValueError(f"No regime weights found in {path}")
        return cls(alpha=alpha, lambdas=lambdas, phi_bar=phi_bar, intercept=intercept)

    def to_dict(self) -> dict[str, Any]:
        regimes = {}
        for regime, weights in self.alpha.items():
            regimes[regime] = {
                "operators": list(weights.keys()),
                "alpha": dict(weights),
                "lambda": dict(self.lambdas.get(regime, {})),
                "phi_bar": dict(self.phi_bar.get(regime, {})),
                "intercept": self.intercept.get(regime, 0.0),
            }
        return {"regimes": regimes}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    def operator_weights(
        self,
        regime: str,
        features: dict[str, float],
        operators: Optional[Iterable[str]] = None,
    ) -> dict[str, float]:
        """w_k(q) = α_k + λ_k · (φ_k − φ̄_k)."""
        beta = float(features["beta"])
        nu = float(features["nu"])
        kappa = float(features["kappa"])
        active = list(operators) if operators is not None else list(self.alpha[regime])
        weights: dict[str, float] = {}
        for op in active:
            alpha = float(self.alpha.get(regime, {}).get(op, 0.0))
            lam = float(self.lambdas.get(regime, {}).get(op, 0.0))
            phi_mean = float(self.phi_bar.get(regime, {}).get(op, 0.0))
            phi = paired_feature(op, beta, nu, kappa)
            weights[op] = alpha + lam * (phi - phi_mean)
        return weights

    def risk(
        self,
        regime: str,
        operator_scores: dict[str, float],
        features: dict[str, float],
        operators: Optional[Iterable[str]] = None,
    ) -> dict[str, Any]:
        active = list(operators) if operators is not None else [
            op for op in self.alpha.get(regime, {}) if op in operator_scores
        ]
        weights = self.operator_weights(regime, features, active)
        logit = float(self.intercept.get(regime, 0.0))
        terms = {}
        for op in active:
            score = float(operator_scores.get(op, 0.0))
            w = float(weights.get(op, 0.0))
            terms[op] = {"score": score, "weight": w, "contribution": w * score}
            logit += w * score
        return {
            "risk": sigmoid(logit),
            "logit": logit,
            "weights": weights,
            "intercept": self.intercept.get(regime, 0.0),
            "terms": terms,
        }
