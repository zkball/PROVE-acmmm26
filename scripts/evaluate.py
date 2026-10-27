#!/usr/bin/env python3
"""Compute AUROC (and a simple threshold sweep) from scored JSONL.

    python scripts/evaluate.py --pred outputs/scored.jsonl
    python scripts/evaluate.py --pred outputs/scored.jsonl --label-key hallu_label
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def auroc(scores: list[float], labels: list[int]) -> dict:
    paired = [(s, y) for s, y in zip(scores, labels) if s is not None and y is not None]
    if not paired:
        return {"auroc": None, "n": 0}
    paired.sort(key=lambda x: -x[0])
    n_pos = sum(1 for _, y in paired if y == 1)
    n_neg = sum(1 for _, y in paired if y == 0)
    if n_pos == 0 or n_neg == 0:
        return {"auroc": 0.5, "n": len(paired), "n_pos": n_pos, "n_neg": n_neg}

    tp = fp = 0
    auc = 0.0
    prev_fpr = prev_tpr = 0.0
    best = {"f1": -1.0}
    for score, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_fpr, prev_tpr = fpr, tpr
        fn = n_pos - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / n_pos
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best["f1"]:
            best = {
                "threshold": score,
                "f1": f1,
                "precision": prec,
                "recall": rec,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": n_neg - fp,
            }
    return {
        "auroc": round(auc, 4),
        "n": len(paired),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "optimal": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in best.items()},
    }


def load_rows(path: Path, score_key: str, label_key: str) -> tuple[list[float], list[int], list[dict]]:
    scores, labels, rows = [], [], []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get(score_key) is None:
                continue
            label = row.get(label_key)
            if label is None:
                continue
            scores.append(float(row[score_key]))
            labels.append(int(label))
            rows.append(row)
    return scores, labels, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PROVE scores")
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--score-key", default="risk")
    parser.add_argument("--label-key", default="label", help="Also accepts hallu_label if present")
    args = parser.parse_args()

    # Prefer --label-key; fall back to hallu_label.
    scores, labels, rows = load_rows(args.pred, args.score_key, args.label_key)
    if not rows:
        scores, labels, rows = load_rows(args.pred, args.score_key, "hallu_label")
    overall = auroc(scores, labels)
    print("overall", json.dumps(overall))

    by_regime: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_regime.setdefault(str(row.get("regime", "unknown")), []).append(i)
    for regime, idxs in sorted(by_regime.items()):
        sub = auroc([scores[i] for i in idxs], [labels[i] for i in idxs])
        print(f"regime={regime}", json.dumps(sub))


if __name__ == "__main__":
    main()
