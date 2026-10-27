"""OpenAI-compatible VLM client for black-box text and vision calls.

Credentials come from the environment. No keys are hardcoded.

    PROVE_API_KEY / OPENAI_API_KEY
    PROVE_BASE_URL / OPENAI_BASE_URL
    PROVE_MODEL
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

from .utils import guess_mime


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


class VLMClient:
    """Thin wrapper around an OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: int = 3,
    ):
        self.model = model or _env("PROVE_MODEL", "OPENAI_MODEL", default="gpt-4o")
        self.api_key = api_key or _env("PROVE_API_KEY", "OPENAI_API_KEY")
        self.base_url = (
            base_url or _env("PROVE_BASE_URL", "OPENAI_BASE_URL") or None
        )
        self.timeout = timeout or float(_env("PROVE_TIMEOUT", default="120"))
        self.max_retries = max_retries
        if not self.api_key:
            raise RuntimeError(
                "Missing API key. Set PROVE_API_KEY or OPENAI_API_KEY "
                "(see .env.example)."
            )
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def call_text(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        return self._complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def call_vision(
        self,
        prompt: str,
        image_path: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        mime = guess_mime(str(path))
        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            },
        ]
        return self._complete(
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _complete(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = resp.choices[0].message.content
                if content is None or not str(content).strip():
                    raise RuntimeError("Empty model response")
                return str(content)
            except Exception as exc:
                last_error = exc
                time.sleep(0.8 * (2 ** attempt))
        raise RuntimeError(f"VLM call failed after {self.max_retries} retries: {last_error}")
