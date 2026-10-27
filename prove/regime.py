"""Stage 1: evidence-structure regime classification and operator routing."""

from __future__ import annotations

from typing import Any, Optional

from . import prompts
from .api import VLMClient
from .utils import parse_json_from_llm

REGIME_NAMES = ("perceptual", "attribute", "compositional")

# Paper Table 1: which operators fire in each regime.
REGIME_OPERATORS: dict[str, tuple[str, ...]] = {
    "perceptual": ("consistency", "contrast", "grounding"),
    "attribute": ("contrast", "grounding", "sufficiency"),
    "compositional": ("contrast", "grounding", "sufficiency", "claim_decomp"),
}

ALL_OPERATORS: tuple[str, ...] = (
    "consistency",
    "contrast",
    "grounding",
    "sufficiency",
    "claim_decomp",
)


def operators_for_regime(regime: str) -> tuple[str, ...]:
    if regime not in REGIME_OPERATORS:
        raise KeyError(f"Unknown regime '{regime}'. Expected one of {REGIME_NAMES}.")
    return REGIME_OPERATORS[regime]


def classify_regime(
    client: VLMClient,
    question: str,
    answer: str,
) -> dict[str, Any]:
    """Zero-shot, text-only regime classifier (supplement Section A.1).

    Temperature is 0. No image is sent. On parse failure the sample is assigned
    to perceptual (the most common regime) and `parse_success` is False.
    """
    prompt = prompts.REGIME_PROMPT.format(question=question, answer=answer)
    raw = client.call_text(prompt, temperature=0.0, max_tokens=512)
    parsed = parse_json_from_llm(raw) or {}
    regime = str(parsed.get("regime", "")).strip().lower()
    if regime not in REGIME_NAMES:
        # Tolerate "Perceptual Regime" / "tier 1" style answers.
        for name in REGIME_NAMES:
            if name in regime:
                regime = name
                break
    parse_success = regime in REGIME_NAMES
    if not parse_success:
        regime = "perceptual"
    return {
        "regime": regime,
        "depth": parsed.get("depth"),
        "entity_count": parsed.get("entity_count"),
        "decomposable": parsed.get("decomposable"),
        "rationale": parsed.get("rationale", ""),
        "parse_success": parse_success,
        "raw": raw,
    }


def maybe_override_regime(regime: Optional[str]) -> Optional[str]:
    if regime is None:
        return None
    regime = str(regime).strip().lower()
    if not regime:
        return None
    aliases = {
        "tier1": "perceptual",
        "tier 1": "perceptual",
        "tier2": "attribute",
        "tier 2": "attribute",
        "tier3": "compositional",
        "tier 3": "compositional",
        "perc": "perceptual",
        "attr": "attribute",
        "comp": "compositional",
    }
    regime = aliases.get(regime, regime)
    if regime not in REGIME_NAMES:
        raise ValueError(f"Unknown regime '{regime}'")
    return regime
