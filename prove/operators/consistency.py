"""Consistency operator (op_cs).

1. Generate two question rephrasings (text, temperature = 0).
2. Re-ask each rephrasing on the image (vision, temperature = 0).
3. Judge semantic equivalence against the baseline (text).

Score = fraction of rephrased answers judged non-equivalent.
"""

from __future__ import annotations

from .. import prompts
from ..api import VLMClient
from ..utils import first_line, parse_json_from_llm
from .base import OperatorResult, failed_result


class ConsistencyOperator:
    name = "consistency"

    def __init__(self, client: VLMClient, n_rephrasings: int = 2):
        self.client = client
        self.n_rephrasings = n_rephrasings

    def _generate_rephrasings(self, question: str) -> list[str]:
        prompt = prompts.REPHRASE_GENERATION_PROMPT.format(question=question)
        raw = self.client.call_text(prompt, temperature=0.0, max_tokens=512)
        parsed = parse_json_from_llm(raw) or {}
        items = parsed.get("rephrasings") or parsed.get("_list") or []
        rephrasings = [str(x).strip() for x in items if str(x).strip()]
        return rephrasings[: self.n_rephrasings]

    def _ask(self, image_path: str, rephrased_question: str) -> str:
        prompt = prompts.REPHRASE_ASK_PROMPT.format(
            rephrased_question=rephrased_question
        )
        raw = self.client.call_vision(
            prompt, image_path, temperature=0.0, max_tokens=256
        )
        parsed = parse_json_from_llm(raw)
        if parsed and parsed.get("answer") is not None:
            return str(parsed["answer"]).strip()
        return first_line(raw)

    def _equivalent(self, question: str, answer_a: str, answer_b: str) -> tuple[bool, str]:
        prompt = prompts.EQUIVALENCE_JUDGE_PROMPT.format(
            question=question, answer_a=answer_a, answer_b=answer_b
        )
        raw = self.client.call_text(prompt, temperature=0.0, max_tokens=256)
        parsed = parse_json_from_llm(raw)
        if parsed and "equivalent" in parsed:
            return bool(parsed["equivalent"]), str(parsed.get("reasoning", ""))
        return True, "parse_failure_lean_equivalent"

    def run(self, image_path: str, question: str, answer: str) -> OperatorResult:
        try:
            rephrasings = self._generate_rephrasings(question)
        except Exception as exc:
            return failed_result(self.name, str(exc), "rephrase_generation")

        if not rephrasings:
            return failed_result(self.name, "No rephrasings generated", "rephrase_generation")

        equivalence = []
        for rephrased in rephrasings:
            try:
                rp_answer = self._ask(image_path, rephrased)
            except Exception as exc:
                equivalence.append({
                    "rephrasing": rephrased,
                    "rephrased_answer": "",
                    "equivalent": True,
                    "reasoning": f"ask_failed: {exc}",
                })
                continue
            try:
                eq, reason = self._equivalent(question, answer, rp_answer)
            except Exception as exc:
                eq, reason = True, f"judge_failed: {exc}"
            equivalence.append({
                "rephrasing": rephrased,
                "rephrased_answer": rp_answer,
                "equivalent": eq,
                "reasoning": reason,
            })

        judged = [row for row in equivalence if row.get("rephrased_answer")]
        if not judged:
            return failed_result(self.name, "All rephrase asks failed", "rephrase_ask")

        n_neq = sum(1 for row in judged if not row["equivalent"])
        score = n_neq / len(judged)
        return OperatorResult(
            name=self.name,
            score=score,
            predicted_label=int(score > 0),
            parse_success=True,
            details={
                "baseline_answer": answer,
                "n_rephrasings": len(judged),
                "n_not_equivalent": n_neq,
                "consistency_rate": 1.0 - score,
                "rounds": equivalence,
            },
        )
