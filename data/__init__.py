"""Local dataset roots for the four public medical VQA benchmarks.

Download the official releases (see README), then place converted JSONL and
images under these paths. Large files stay local and are gitignored.

Expected layout after preparation:

    data/
      slake/images/          xmlab*/source.jpg
      slake/{train,val,test}.jsonl
      vqa_rad/images/
      vqa_rad/{train,test}.jsonl
      vqa_med_2019/images/
      vqa_med_2019/{train,val,test}.jsonl
      pmc_vqa/images/
      pmc_vqa/{train,test}.jsonl

Each JSONL line is:
    {"image_path": "<relative to images/>", "question": "...", "answer": "..."}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

DATA_ROOT = Path(__file__).resolve().parent

DATASETS: dict[str, dict] = {
    "slake": {
        "name": "SLAKE",
        "dir": DATA_ROOT / "slake",
        "images": DATA_ROOT / "slake" / "images",
        "splits": ("train", "val", "test"),
        "project_page": "https://www.med-vqa.com/slake/",
        "download": "https://huggingface.co/datasets/BoKelvin/SLAKE",
        "note": "Use English QA only (q_lang=en). Paper test split: 1,061 English items.",
    },
    "vqa_rad": {
        "name": "VQA-RAD",
        "dir": DATA_ROOT / "vqa_rad",
        "images": DATA_ROOT / "vqa_rad" / "images",
        "splits": ("train", "test"),
        "project_page": "https://osf.io/89kps/",
        "download": "https://huggingface.co/datasets/flaviagiammarino/vqa-rad",
        "note": "Clinician-authored radiology VQA. Paper uses this split for calibration.",
    },
    "vqa_med_2019": {
        "name": "VQA-Med-2019",
        "dir": DATA_ROOT / "vqa_med_2019",
        "images": DATA_ROOT / "vqa_med_2019" / "images",
        "splits": ("train", "val", "test"),
        "project_page": "https://www.imageclef.org/2019/medical/vqa",
        "download": "https://huggingface.co/datasets/claudioreeves/imageclef-vqa-med-2019",
        "note": "ImageCLEF 2019 medical VQA (Modality / Plane / Organ / Abnormality).",
    },
    "pmc_vqa": {
        "name": "PMC-VQA",
        "dir": DATA_ROOT / "pmc_vqa",
        "images": DATA_ROOT / "pmc_vqa" / "images",
        "splits": ("train", "test"),
        "project_page": "https://github.com/xiaoman-zhang/PMC-VQA",
        "download": "https://huggingface.co/datasets/xmcmic/PMC-VQA",
        "note": "Large PMC figure VQA. Optional extra public set; not in the main 8,048-row table.",
    },
}


def dataset_dir(name: str) -> Path:
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Known: {list(DATASETS)}")
    return DATASETS[name]["dir"]


def images_dir(name: str) -> Path:
    return DATASETS[name]["images"]


def jsonl_path(name: str, split: str) -> Path:
    meta = DATASETS[name]
    if split not in meta["splits"]:
        raise ValueError(f"{name} has splits {meta['splits']}, not '{split}'")
    return meta["dir"] / f"{split}.jsonl"


def is_prepared(name: str, split: str = "test") -> bool:
    """True if the split JSONL exists and the images folder is non-empty."""
    path = jsonl_path(name, split)
    img = images_dir(name)
    if not path.exists():
        return False
    has_image = any(img.rglob("*.jpg")) or any(img.rglob("*.png")) or any(img.rglob("*.jpeg"))
    return has_image


def resolve_image(name: str, image_path: str) -> Path:
    """Join a JSONL-relative image path onto data/<name>/images/."""
    root = images_dir(name)
    rel = Path(image_path)
    if rel.is_absolute():
        return rel
    return root / rel


def load_split(
    name: str,
    split: str = "test",
    *,
    limit: Optional[int] = None,
    require_images: bool = False,
) -> list[dict]:
    """Load a prepared split. Adds image_path_resolved and image_exists."""
    path = jsonl_path(name, split)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download {DATASETS[name]['name']} from "
            f"{DATASETS[name]['download']} and place files under {dataset_dir(name)} "
            "(see README Datasets)."
        )
    rows: list[dict] = []
    with path.open() as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            img = resolve_image(name, row.get("image_path") or row.get("image", ""))
            row["image_path_resolved"] = str(img)
            row["image_exists"] = img.exists()
            if require_images and not row["image_exists"]:
                continue
            rows.append(row)
    return rows


def status() -> dict[str, dict]:
    """Print-friendly prepared/missing status for every dataset split."""
    out = {}
    for name, meta in DATASETS.items():
        info = {"name": meta["name"], "dir": str(meta["dir"]), "splits": {}}
        for split in meta["splits"]:
            path = jsonl_path(name, split)
            n = None
            if path.exists():
                with path.open() as f:
                    n = sum(1 for line in f if line.strip())
            info["splits"][split] = {
                "jsonl": str(path),
                "exists": path.exists(),
                "n": n,
                "prepared": is_prepared(name, split) if path.exists() else False,
            }
        out[name] = info
    return out


if __name__ == "__main__":
    for name, info in status().items():
        print(f"{name} ({info['name']}) -> {info['dir']}")
        for split, s in info["splits"].items():
            flag = "ready" if s["prepared"] else ("jsonl only" if s["exists"] else "missing")
            n = s["n"] if s["n"] is not None else "-"
            print(f"  {split:5s}  {flag:12s}  n={n}")
