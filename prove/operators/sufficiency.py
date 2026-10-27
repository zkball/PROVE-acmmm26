"""Sufficiency operator (op_sf).

Ask whether the image can support the given answer, K times
(vision, temperature = 0.7). Score = n_insufficient / K.
"""

from __future__ import annotations

from .. import prompts
from ..api import VLMClient
from ..utils import parse_json_from_llm
from .base import OperatorResult, failed_result


class SufficiencyOperator:
    name = "sufficiency"

    def __init__(self, client: VLMClient, k: int = 3, vote_temperature: float = 0.7):
        self.client = client
        self.k = max(1, k)
        self.vote_temperature = vote_temperature

    def run(self, image_path: str, question: str, answer: str) -> OperatorResult:
        prompt = prompts.SUFFICIENCY_PROMPT.format(
            question=question, baseline_answer=answer
        )
        votes = []
        for _ in range(self.k):
            temp = 0.0 if self.k == 1 else self.vote_temperature
            try:
                raw = self.client.call_vision(
                    prompt, image_path, temperature=temp, max_tokens=512
                )
            except Exception:
                continue
            parsed = parse_json_from_llm(raw)
            if parsed and "sufficient" in parsed:
                votes.append({
                    "sufficient": bool(parsed["sufficient"]),
                    "missing_evidence": parsed.get("missing_evidence", ""),
                    "reasoning": parsed.get("reasoning", ""),
                })

        if not votes:
            return failed_result(self.name, "All sufficiency rounds failed", "sufficiency")

        n_insufficient = sum(1 for v in votes if not v["sufficient"])
        score = n_insufficient / len(votes)
        return OperatorResult(
            name=self.name,
            score=score,
            predicted_label=int(score > 0.5),
            parse_success=True,
            details={
                "baseline_answer": answer,
                "num_rounds": self.k,
                "n_votes": len(votes),
                "n_insufficient": n_insufficient,
                "missing_evidence": votes[0].get("missing_evidence", ""),
                "vote_details": votes,
            },
        )
