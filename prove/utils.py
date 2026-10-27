"""Shared helpers: JSON parsing, image MIME types, tokenization."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)?")


def guess_mime(image_path: str) -> str:
    """Return the MIME type from the file extension.

    Claude/Bedrock rejects a JPEG sent as image/png. Never hardcode image/png.
    """
    ext = Path(image_path).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }.get(ext, "image/jpeg")


def tokenize(text: str) -> list[str]:
    """Lowercased alphanumeric tokens (hyphenated compounds kept together)."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def parse_json_from_llm(text: str) -> Optional[dict[str, Any]]:
    """Extract the first JSON object or array from an LLM response."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[1:end]).strip()
        if text.lower().startswith("json"):
            text = text[4:].lstrip()

    for start_ch, end_ch in (("{", "}"), ("[", "]")):
        start = text.find(start_ch)
        end = text.rfind(end_ch) + 1
        if start >= 0 and end > start:
            snippet = text[start:end]
            try:
                parsed = json.loads(snippet)
            except json.JSONDecodeError:
                snippet = _repair_booleans(snippet)
                try:
                    parsed = json.loads(snippet)
                except json.JSONDecodeError:
                    continue
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return {"_list": parsed}
    return None


def _repair_booleans(text: str) -> str:
    """Replace bare true/false-like tokens that some models emit unquoted."""
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)
    return text


def first_line(text: str) -> str:
    if not text:
        return ""
    return text.strip().split("\n")[0].strip()
