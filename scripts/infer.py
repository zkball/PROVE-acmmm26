#!/usr/bin/env python3
"""Run PROVE on one sample or a JSONL batch.

Single sample:
    python scripts/infer.py --image img.jpg --question "What modality is this?" --answer CT

Batch:
    python scripts/infer.py --input examples/sample.jsonl --output outputs/scored.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prove.api import VLMClient
from prove.calibration import Calibration
from prove.pipeline import PROVE
from prove.regime import ALL_OPERATORS


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _row_fields(row: dict) -> tuple[str, str, str, str]:
    sample_id = str(row.get("sample_id") or row.get("id") or "")
    image = row.get("image") or row.get("image_path") or row.get("image_path_resolved")
    question = row.get("question")
    answer = row.get("answer") or row.get("model_answer") or row.get("baseline_answer")
    if not image or not question or answer is None:
        raise ValueError(
            "Each row needs image/image_path, question, and answer/model_answer. "
            f"Got keys: {sorted(row)}"
        )
    return sample_id, str(image), str(question), str(answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PROVE hallucination detection")
    parser.add_argument("--image", help="Path to a medical image")
    parser.add_argument("--question", help="Clinical question")
    parser.add_argument("--answer", help="VLM answer to verify")
    parser.add_argument("--input", type=Path, help="JSONL with image/question/answer")
    parser.add_argument("--output", type=Path, help="Write JSONL scores here")
    parser.add_argument("--weights", type=Path, default=ROOT / "configs" / "paper_weights.json")
    parser.add_argument("--k", type=int, default=3, help="Voting rounds (paper uses 3)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel operators per sample")
    parser.add_argument("--model", help="Override PROVE_MODEL")
    parser.add_argument("--regime", help="Skip the classifier and force a regime")
    parser.add_argument(
        "--operators",
        help="Comma-separated operator list, overriding regime routing",
    )
    parser.add_argument("--no-details", action="store_true", help="Drop bulky operator traces")
    parser.add_argument("--resume", action="store_true", help="Skip sample_ids already in --output")
    parser.add_argument("--limit", type=int, default=0, help="Max batch rows (0 = all)")
    args = parser.parse_args()

    if not args.image and not args.input:
        parser.error("Provide --image/--question/--answer or --input JSONL")
    if args.image and (not args.question or args.answer is None):
        parser.error("Single-sample mode needs --question and --answer")

    calib = Calibration.from_json(args.weights) if args.weights.exists() else Calibration.paper_default()
    client_kwargs = {}
    if args.model:
        client_kwargs["model"] = args.model
    detector = PROVE(
        client=VLMClient(**client_kwargs),
        calibration=calib,
        k=args.k,
        workers=args.workers,
    )
    op_override = None
    if args.operators:
        op_override = [x.strip() for x in args.operators.split(",") if x.strip()]
        unknown = [x for x in op_override if x not in ALL_OPERATORS]
        if unknown:
            parser.error(f"Unknown operators {unknown}. Known: {ALL_OPERATORS}")

    if args.image:
        result = detector.score(
            image=args.image,
            question=args.question or "",
            answer=args.answer or "",
            regime=args.regime,
            operators=op_override,
            include_details=not args.no_details,
        )
        text = json.dumps(result, ensure_ascii=False, indent=2)
        print(text)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n")
        return

    rows = _load_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]

    done = set()
    outfile = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.resume and args.output.exists():
            for line in args.output.read_text().splitlines():
                if line.strip():
                    done.add(json.loads(line).get("sample_id"))
            outfile = args.output.open("a")
        else:
            outfile = args.output.open("w")

    n_ok = 0
    n_fail = 0
    try:
        for i, row in enumerate(rows, start=1):
            sample_id, image, question, answer = _row_fields(row)
            if sample_id and sample_id in done:
                continue
            try:
                result = detector.score(
                    image=image,
                    question=question,
                    answer=answer,
                    sample_id=sample_id,
                    regime=args.regime or row.get("regime"),
                    operators=op_override,
                    include_details=not args.no_details,
                )
            except Exception as exc:
                n_fail += 1
                result = {
                    "sample_id": sample_id,
                    "question": question,
                    "answer": answer,
                    "image": image,
                    "error": str(exc),
                    "risk": None,
                }
            if "label" in row:
                result["label"] = row["label"]
            elif "hallu_label" in row:
                result["label"] = row["hallu_label"]
            line = json.dumps(result, ensure_ascii=False)
            print(line)
            if outfile:
                outfile.write(line + "\n")
                outfile.flush()
            if result.get("risk") is not None:
                n_ok += 1
            print(f"  [{i}/{len(rows)}] id={sample_id} regime={result.get('regime')} risk={result.get('risk')}", file=sys.stderr)
    finally:
        if outfile:
            outfile.close()
    print(f"Done. ok={n_ok} fail={n_fail}", file=sys.stderr)


if __name__ == "__main__":
    main()
