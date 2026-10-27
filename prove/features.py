"""Deterministic proof features β, ν, κ (paper Eqs. 1–3).

No LLM calls. Features are computed from the question-answer text alone.
"""

from __future__ import annotations

import math
import re
from typing import Any

from .lexicon import VISUAL_LEXICON
from .utils import tokenize


_YESNO_ANSWER = {
    "yes", "no", "y", "n", "true", "false", "present", "absent",
    "normal", "abnormal",
}

_YESNO_PREFIX = re.compile(
    r"^\s*(is|are|was|were|do|does|did|can|could|has|have|had)\b",
    re.I,
)

_CLAUSE_WORD = re.compile(r"\b(and|or|with|which|that|but)\b", re.I)


def estimate_answer_set_size(question: str, answer: str) -> int:
    """Approximate |A_q| from question type (paper: training-set statistics).

    Examples from the paper: yes/no → 2, modality → ~6, organ ID → ~20,
    open-ended diagnosis → ~200.
    """
    q = (question or "").lower()
    a = (answer or "").strip().lower()

    if a in _YESNO_ANSWER or _YESNO_PREFIX.search(q) or "yes or no" in q or "yes/no" in q:
        return 2
    if any(k in q for k in (
        "normal or abnormal", "normal/abnormal", "normal vs",
        "which side", "which hemisphere", "left or right", "right or left",
        "with or without contrast",
    )):
        return 2
    if any(k in q for k in (
        "modality", "modalities", "imaging technique", "type of scan",
        "type of image", "taken with", "acquired with", "kind of image",
    )):
        return 6
    if any(k in q for k in (
        "which plane", "what plane", "imaging plane", "which view",
        "what view", "projection", "orientation",
    )):
        return 6
    if any(k in q for k in (
        "what organ", "which organ", "which structure", "what structure",
        "which part", "what part of the body", "anatomical structure",
        "body part", "organs are", "organs visible",
    )):
        return 20
    if any(k in q for k in (
        "describe", "what is wrong", "what are the", "what abnormalities",
        "what findings", "explain", "what do you see", "impression",
        "diagnosis",
    )):
        return 200
    if len(a.split()) >= 8:
        return 200
    return 20


def _answer_boundedness(question: str, answer: str) -> float:
    """β = 1/2 [ 1/log2(|A|+1) + 1/(1+|a|_tokens) ]."""
    card = max(1, estimate_answer_set_size(question, answer))
    beta_form = 1.0 / math.log2(card + 1)
    n_tokens = max(0, len(tokenize(answer)))
    beta_length = 1.0 / (1.0 + n_tokens)
    return 0.5 * (beta_form + beta_length)


def _visual_term_count(question: str) -> int:
    """|tokens(q) ∩ L_visual| as in paper Eq. 3."""
    return len(set(tokenize(question)) & VISUAL_LEXICON)


def _visual_specificity(question: str) -> float:
    """ν = 1 - 1/(1 + |tokens(q) ∩ L_visual|)."""
    n_visual = _visual_term_count(question)
    return 1.0 - 1.0 / (1.0 + n_visual)


def _n_clauses(answer: str) -> int:
    """Count clause boundaries: commas, semicolons, extra periods, conjunctions.

    A trailing sentence period is not counted. ', and' is one boundary, not two.
    """
    if not answer or not answer.strip():
        return 0
    text = answer.strip()
    n_punct = len(re.findall(r"[,;]", text))
    # Periods that are not the final character.
    n_period = len(re.findall(r"\.(?!\s*$)", text))
    # Conjunctions not immediately preceded by a comma (avoid ', and').
    n_conj = 0
    for match in _CLAUSE_WORD.finditer(text):
        prefix = text[:match.start()].rstrip()
        if prefix.endswith(","):
            continue
        n_conj += 1
    return n_punct + n_period + n_conj


def _claim_complexity(answer: str) -> float:
    """κ = 1 - 1/(1 + n_clauses(a))."""
    return 1.0 - 1.0 / (1.0 + _n_clauses(answer))


def compute_proof_features(question: str, answer: str) -> dict[str, Any]:
    """Return β, ν, κ and the intermediate counts used to compute them."""
    card = estimate_answer_set_size(question, answer)
    n_visual = _visual_term_count(question)
    n_clauses = _n_clauses(answer)
    beta = _answer_boundedness(question, answer)
    nu = _visual_specificity(question)
    kappa = _claim_complexity(answer)
    return {
        "beta": round(beta, 6),
        "nu": round(nu, 6),
        "kappa": round(kappa, 6),
        "answer_set_size": card,
        "n_answer_tokens": len(tokenize(answer)),
        "n_visual_terms": n_visual,
        "n_clauses": n_clauses,
    }


# Operator → paired proof feature (paper Table 1).
FEATURE_PAIRING = {
    "consistency": lambda beta, nu, kappa: 0.5 * (beta + kappa),
    "contrast": lambda beta, nu, kappa: beta,
    "grounding": lambda beta, nu, kappa: nu,
    "sufficiency": lambda beta, nu, kappa: 1.0 - nu,
    "claim_decomp": lambda beta, nu, kappa: kappa,
}


def paired_feature(operator: str, beta: float, nu: float, kappa: float) -> float:
    if operator not in FEATURE_PAIRING:
        raise KeyError(f"No proof-feature pairing for operator '{operator}'")
    return FEATURE_PAIRING[operator](beta, nu, kappa)
