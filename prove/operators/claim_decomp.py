"""Claim decomposition operator (op_cd).

1. Decompose the answer into atomic sub-claims (text).
2. Verify each sub-claim independently against the image (vision).

Score = fraction of sub-claims judged unsupported.
"""

from __future__ import annotations

from .. import prompts
from ..api import VLMClient
from ..utils import parse_json_from_llm
from .base import OperatorResult, failed_result


class ClaimDecompOperator:
    name = "claim_decomp"

    def __init__(self, client: VLMClient, failure_threshold: float = 0.5):
        self.client = client
        self.failure_threshold = failure_threshold

    def run(self, image_path: str, question: str, answer: str) -> OperatorResult:
        decompose_prompt = prompts.DECOMPOSE_PROMPT.format(
            question=question, baseline_answer=answer
        )
        try:
            raw1 = self.client.call_text(
                decompose_prompt, temperature=0.0, max_tokens=512
            )
        except Exception as exc:
            return failed_result(self.name, str(exc), "decompose")

        parsed1 = parse_json_from_llm(raw1) or {}
        claims = parsed1.get("claims") or parsed1.get("atomic_claims") or parsed1.get("_list") or []
        claims = [str(c).strip() for c in claims if str(c).strip()]
        if not claims:
            return failed_result(self.name, "Could not extract claims list", "decompose")

        claim_results = []
        for claim in claims:
            verify_prompt = prompts.VERIFY_CLAIM_PROMPT.format(
                question=question, claim=claim
            )
            try:
                raw = self.client.call_vision(
                    verify_prompt, image_path, temperature=0.0, max_tokens=256
                )
            except Exception as exc:
                claim_results.append({"claim": claim, "supported": None, "error": str(exc)})
                continue
            parsed = parse_json_from_llm(raw)
            if parsed and "supported" in parsed:
                claim_results.append({
                    "claim": claim,
                    "supported": bool(parsed["supported"]),
                    "reasoning": parsed.get("reasoning", ""),
                })
            else:
                claim_results.append({"claim": claim, "supported": None, "parse_error": True})

        verified = [c for c in claim_results if c.get("supported") is not None]
        if not verified:
            return failed_result(self.name, "No claims could be verified", "verify")

        n_failed = sum(1 for c in verified if not c["supported"])
        score = n_failed / len(verified)
        return OperatorResult(
            name=self.name,
            score=score,
            predicted_label=int(score >= self.failure_threshold),
            parse_success=True,
            details={
                "claims": claims,
                "claim_results": claim_results,
                "n_claims": len(claims),
                "n_verified": len(verified),
                "n_failed": n_failed,
                "failure_rate": round(score, 4),
            },
        )
