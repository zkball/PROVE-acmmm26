"""Contrast operator (op_ct).

1. Independent answer from the verifier (vision, temperature = 0).
2. Compare against the baseline K times (text, temperature = 0.7).

Score = n_disagree / K.
"""

from __future__ import annotations

from .. import prompts
from ..api import VLMClient
from ..utils import first_line, parse_json_from_llm
from .base import OperatorResult, failed_result


class ContrastOperator:
    name = "contrast"

    def __init__(self, client: VLMClient, k: int = 3, vote_temperature: float = 0.7):
        self.client = client
        self.k = max(1, k)
        self.vote_temperature = vote_temperature

    def run(self, image_path: str, question: str, answer: str) -> OperatorResult:
        prompt1 = prompts.INDEPENDENT_ANSWER_PROMPT.format(question=question)
        try:
            raw1 = self.client.call_vision(
                prompt1, image_path, temperature=0.0, max_tokens=256
            )
        except Exception as exc:
            return failed_result(self.name, str(exc), "independent_answer")

        parsed1 = parse_json_from_llm(raw1)
        independent = (
            str(parsed1["answer"]).strip()
            if parsed1 and parsed1.get("answer") is not None
            else first_line(raw1)
        )
        if not independent:
            return failed_result(self.name, "Empty independent answer", "independent_answer")

        prompt2 = prompts.CONTRAST_COMPARISON_PROMPT.format(
            question=question,
            independent_answer=independent,
            baseline_answer=answer,
        )
        votes = []
        for _ in range(self.k):
            temp = 0.0 if self.k == 1 else self.vote_temperature
            try:
                raw = self.client.call_text(prompt2, temperature=temp, max_tokens=256)
            except Exception:
                continue
            parsed = parse_json_from_llm(raw)
            if parsed and "agree" in parsed:
                votes.append({
                    "agree": bool(parsed["agree"]),
                    "reasoning": parsed.get("reasoning", ""),
                })

        if not votes:
            return failed_result(self.name, "All comparison rounds failed", "comparison")

        n_disagree = sum(1 for v in votes if not v["agree"])
        score = n_disagree / len(votes)
        return OperatorResult(
            name=self.name,
            score=score,
            predicted_label=int(score > 0.5),
            parse_success=True,
            details={
                "independent_answer": independent,
                "baseline_answer": answer,
                "num_rounds": self.k,
                "n_votes": len(votes),
                "n_disagree": n_disagree,
                "vote_details": votes,
            },
        )
