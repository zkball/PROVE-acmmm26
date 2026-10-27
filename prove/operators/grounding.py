"""Grounding operator (op_gr).

1. Elicit visual evidence without answering (vision, temperature = 0).
2. Judge whether the evidence supports the baseline K times (text, 0.7).

Score = n_unsupported / K.
"""

from __future__ import annotations

from .. import prompts
from ..api import VLMClient
from ..utils import parse_json_from_llm
from .base import OperatorResult, failed_result


class GroundingOperator:
    name = "grounding"

    def __init__(self, client: VLMClient, k: int = 3, vote_temperature: float = 0.7):
        self.client = client
        self.k = max(1, k)
        self.vote_temperature = vote_temperature

    def run(self, image_path: str, question: str, answer: str) -> OperatorResult:
        evidence_prompt = prompts.EVIDENCE_PROMPT.format(question=question)
        try:
            evidence = self.client.call_vision(
                evidence_prompt, image_path, temperature=0.0, max_tokens=512
            )
        except Exception as exc:
            return failed_result(self.name, str(exc), "evidence")

        judge_prompt = prompts.GROUNDING_JUDGE_PROMPT.format(
            question=question,
            baseline_answer=answer,
            evidence=evidence,
        )
        votes = []
        for _ in range(self.k):
            temp = 0.0 if self.k == 1 else self.vote_temperature
            try:
                raw = self.client.call_text(judge_prompt, temperature=temp, max_tokens=256)
            except Exception:
                continue
            parsed = parse_json_from_llm(raw)
            if parsed and "supported" in parsed:
                votes.append({
                    "supported": bool(parsed["supported"]),
                    "reasoning": parsed.get("reasoning", ""),
                })

        if not votes:
            return failed_result(self.name, "All judge rounds failed", "judge")

        n_unsupported = sum(1 for v in votes if not v["supported"])
        score = n_unsupported / len(votes)
        return OperatorResult(
            name=self.name,
            score=score,
            predicted_label=int(score > 0.5),
            parse_success=True,
            details={
                "evidence": evidence,
                "baseline_answer": answer,
                "num_rounds": self.k,
                "n_votes": len(votes),
                "n_not_supported": n_unsupported,
                "vote_details": votes,
            },
        )
