"""PROVE: Proof-guided Regime-aware Operator Verification.

Black-box hallucination detector for medical visual question answering.
"""

from .calibration import Calibration
from .features import compute_proof_features
from .pipeline import PROVE
from .regime import REGIME_OPERATORS, classify_regime

__version__ = "0.1.0"
__all__ = [
    "PROVE",
    "Calibration",
    "compute_proof_features",
    "classify_regime",
    "REGIME_OPERATORS",
]
