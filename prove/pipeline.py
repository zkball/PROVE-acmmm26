"""PROVE inference pipeline (paper Algorithm 1).

Stage 1  Classify the question into a verification regime (text-only).
Stage 2  Run the regime-specific operator subset (K-round black-box queries).
Stage 3  Aggregate operator scores with adaptive calibration into risk in [0, 1].
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from .api import VLMClient
from .calibration import Calibration
from .features import compute_proof_features
from .operators import (
    ClaimDecompOperator,
    ConsistencyOperator,
    ContrastOperator,
    GroundingOperator,
    SufficiencyOperator,
)
from .operators.base import OperatorResult
from .regime import (
    ALL_OPERATORS,
    classify_regime,
    maybe_override_regime,
    operators_for_regime,
)


class PROVE:
    """Black-box hallucination detector for medical VQA."""

    def __init__(
        self,
        client: Optional[VLMClient] = None,
        calibration: Optional[Calibration] = None,
        k: int = 3,
        vote_temperature: float = 0.7,
        workers: int = 4,
    ):
        self.client = client or VLMClient()
        self.calibration = calibration or Calibration.paper_default()
        self.k = k
        self.vote_temperature = vote_temperature
        self.workers = workers
        self._operators = {
            "consistency": ConsistencyOperator(self.client),
            "contrast": ContrastOperator(
                self.client, k=k, vote_temperature=vote_temperature
            ),
            "grounding": GroundingOperator(
                self.client, k=k, vote_temperature=vote_temperature
            ),
            "sufficiency": SufficiencyOperator(
                self.client, k=k, vote_temperature=vote_temperature
            ),
            "claim_decomp": ClaimDecompOperator(self.client),
        }

    def score(
        self,
        image: str,
        question: str,
        answer: str,
        sample_id: str = "",
        regime: Optional[str] = None,
        operators: Optional[Sequence[str]] = None,
        include_details: bool = True,
    ) -> dict[str, Any]:
        """Run Algorithm 1 and return a hallucination risk in [0, 1]."""
        image_path = str(Path(image).expanduser())
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        forced = maybe_override_regime(regime)
        if forced is None:
            regime_info = classify_regime(self.client, question, answer)
        else:
            regime_info = {
                "regime": forced,
                "parse_success": True,
                "rationale": "user override",
                "depth": None,
                "entity_count": None,
                "decomposable": None,
                "raw": "",
            }
        regime_name = regime_info["regime"]
        active = tuple(operators) if operators else operators_for_regime(regime_name)
        for name in active:
            if name not in self._operators:
                raise KeyError(f"Unknown operator '{name}'. Known: {list(self._operators)}")

        op_results = self._run_operators(image_path, question, answer, active)
        scores = {name: result.score for name, result in op_results.items()}
        features = compute_proof_features(question, answer)
        calibrated = self.calibration.risk(regime_name, scores, features, active)

        out: dict[str, Any] = {
            "sample_id": sample_id,
            "question": question,
            "answer": answer,
            "image": image_path,
            "regime": regime_name,
            "operators": list(active),
            "operator_scores": scores,
            "features": {
                "beta": features["beta"],
                "nu": features["nu"],
                "kappa": features["kappa"],
            },
            "weights": calibrated["weights"],
            "risk": calibrated["risk"],
            "logit": calibrated["logit"],
            "k": self.k,
            "verifier": self.client.model,
        }
        if include_details:
            out["regime_details"] = {
                k: v for k, v in regime_info.items() if k != "raw"
            }
            out["feature_details"] = features
            out["operator_details"] = {
                name: result.to_dict() for name, result in op_results.items()
            }
            out["calibration"] = {
                "intercept": calibrated["intercept"],
                "terms": calibrated["terms"],
            }
        return out

    def _run_operators(
        self,
        image_path: str,
        question: str,
        answer: str,
        names: Iterable[str],
    ) -> dict[str, OperatorResult]:
        names = list(names)
        if not names:
            return {}
        workers = max(1, min(self.workers, len(names)))
        results: dict[str, OperatorResult] = {}

        def _run(name: str) -> tuple[str, OperatorResult]:
            return name, self._operators[name].run(image_path, question, answer)

        if workers == 1 or len(names) == 1:
            for name in names:
                key, value = _run(name)
                results[key] = value
            return results

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run, name) for name in names]
            for fut in as_completed(futures):
                name, value = fut.result()
                results[name] = value
        return {name: results[name] for name in names}


def available_operators() -> tuple[str, ...]:
    return ALL_OPERATORS
