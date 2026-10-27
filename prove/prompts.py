"""Prompt templates from the PROVE supplementary material (Section A).

All prompts are zero-shot with no in-context examples. Placeholders in curly
braces are filled at runtime. Keep this file aligned with the paper supplement.
"""

# ─── Stage 1: regime classifier (text-only, temperature = 0) ─────────────

REGIME_PROMPT = """You are analyzing the verification structure of a medical VQA question-answer pair. Determine the proof chain required to verify the answer from the image alone.

Analyze three dimensions:

1. OBSERVATION DEPTH. How many independent reasoning steps connect image content to the answer?
   -- Single step: the answer is directly readable from one look at the image (a global property, a present/absent judgment, an immediately recognizable feature).
   -- Two steps: the answer depends on first correctly identifying an entity, then extracting a specific property of that entity. The property cannot be verified without the identification step succeeding first.
   -- Three or more steps: the answer composes multiple independent observations, relates entities spatially, or chains inferences across several image features.

2. ENTITY DEPENDENCY. How many distinct anatomical or visual entities must be correctly identified before the answer can be verified?
   -- Zero or one: the whole image or one obvious structure suffices.
   -- Exactly one, but the answer is conditional on a property of that entity rather than its presence.
   -- Two or more entities, or an entity plus its spatial/relational context.

3. ANSWER DECOMPOSABILITY. Can the answer be broken into independently verifiable sub-claims?
   -- No: the answer is atomic (a single label, yes/no, one name).
   -- No: the answer is a single value conditional on correct entity identification.
   -- Yes: the answer contains multiple claims that could individually be true or false.

Map to one of three regimes:

PERCEPTUAL: depth=1, entities<=1, atomic answer. Hallucinations arise from language priors overriding a single visual observation.

ATTRIBUTE: depth=2, entity=1 (conditional), non-decomposable answer. Hallucinations arise from confident confabulation of a property without grounding it in the identified entity.

COMPOSITIONAL: depth>=3, entities>=2 or decomposable answer. Hallucinations arise from partial correctness where some sub-claims hold and others are fabricated.

Question: {question}
Answer: {answer}

Return ONLY valid JSON:
{{"regime": "perceptual|attribute|compositional",
 "depth": <integer>,
 "entity_count": <integer>,
 "decomposable": true/false,
 "rationale": "one-sentence proof chain analysis"}}"""


# ─── Consistency operator ────────────────────────────────────────────────

REPHRASE_GENERATION_PROMPT = """Generate two semantically equivalent rephrasings of this medical question that test whether the answer is grounded in visual evidence rather than language priors.

Question: {question}

Requirements: (1) Preserve the clinical intent and expected answer. (2) Vary the framing, perspective, or phrasing. (3) Each rephrasing should be self-contained.

Return JSON: {{"rephrasings": ["<r1>", "<r2>"]}}"""

REPHRASE_ASK_PROMPT = """{rephrased_question}
Respond in JSON: {{"answer": "<your answer>"}}"""

EQUIVALENCE_JUDGE_PROMPT = """A model answered a medical image question twice. Are the answers semantically equivalent?

Question: {question}
Answer A: {answer_a}
Answer B: {answer_b}

Synonyms and different specificity levels ARE equivalent. Different entities are NOT equivalent. If in doubt, lean EQUIVALENT.

Respond in JSON: {{"equivalent": true/false, "reasoning": "<brief>"}}"""


# ─── Contrast operator ───────────────────────────────────────────────────

INDEPENDENT_ANSWER_PROMPT = """You are a medical imaging expert. Examine this image and answer the question.

Question: {question}

Give a concise, specific answer.

Respond in JSON: {{"answer": "<your answer>"}}"""

CONTRAST_COMPARISON_PROMPT = """Two experts answered the same medical image question:

Question: {question}
Expert A: {independent_answer}
Expert B: {baseline_answer}

Do these answers agree, identifying the same thing?

Rules: Synonyms and minor wording differences count as AGREE. Different entities, locations, or diagnoses count as DISAGREE. If in doubt, DISAGREE.

Respond in JSON: {{"agree": true/false, "reasoning": "<brief>"}}"""


# ─── Grounding operator ──────────────────────────────────────────────────

EVIDENCE_PROMPT = """You are analyzing a medical image. A question was asked:

Question: {question}

WITHOUT answering the question, describe the key visual features relevant to answering it. Focus on what anatomical structures, findings, and characteristics you observe."""

GROUNDING_JUDGE_PROMPT = """Evaluate whether visual evidence from a medical image supports a proposed answer.

Question: {question}
Proposed answer: {baseline_answer}

Visual evidence from an expert: {evidence}

Does the evidence support the proposed answer? The evidence must specifically confirm the answer, not just be loosely related. Partial correctness is NOT sufficient. When in doubt, NOT supported.

Respond in JSON: {{"supported": true/false, "reasoning": "<brief>"}}"""


# ─── Sufficiency operator ────────────────────────────────────────────────

SUFFICIENCY_PROMPT = """Does this medical image provide sufficient evidence to support the given answer?

Question: {question}
Proposed answer: {baseline_answer}

Consider whether the image type, quality, and visible content can confirm the answer.

Respond in JSON:
{{"sufficient": true/false,
 "missing_evidence": "<what would be needed if insufficient>",
 "reasoning": "<brief>"}}"""


# ─── Claim decomposition operator ────────────────────────────────────────

DECOMPOSE_PROMPT = """Decompose this answer into independent, atomic claims that can each be verified from the image.

Question: {question}
Answer: {baseline_answer}

Each claim should be a single factual assertion, independently verifiable.

Respond in JSON: {{"claims": ["<claim 1>", "<claim 2>", ...]}}"""

VERIFY_CLAIM_PROMPT = """Verify a specific claim about a medical image.

Original question: {question}
Claim to verify: {claim}

Is this claim supported by what you see in the image? Be strict: the claim must be clearly supported by visible evidence.

Respond in JSON: {{"supported": true/false, "reasoning": "<brief>"}}"""
