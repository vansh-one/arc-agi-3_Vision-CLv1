"""Vispark Vision API client with Continual Learning + Tool Calling + Image input.

Single source of truth for all calls to ``POST /model/text/vision``. Handles:

- text + image content (multimodal)
- ``continual_learning=true`` weights round-trip
- tool calling (function-spec list)
- streaming disabled (we want a deterministic blob per call)
- retries with exponential backoff for 429 / 5xx / network errors
- ``insufficient tokens`` recognition (used as a training stop signal)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

log = logging.getLogger(__name__)

VisionSize = Literal["small", "medium", "large"]

DEFAULT_URL = "https://api.lab.vispark.in/model/text/vision"
DEFAULT_TIMEOUT = 180.0
MAX_RETRIES = 5
INSUFFICIENT_TOKEN_MARKERS = ("insufficient", "exceed", "too long", "context limit")


class VisionAPIError(RuntimeError):
    """Generic Vision API failure."""


class VisionInsufficientTokensError(VisionAPIError):
    """Raised when the Vision API rejects the request because the CL weights blob
    or the user prompt has saturated the 980 K-token window.

    Used as one of the three training stop criteria.
    """


@dataclass
class VisionToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class VisionResponse:
    """Normalised response from one Vision call."""

    type: Literal["text", "tool_calls"]
    content: str = ""
    tool_calls: list[VisionToolCall] = field(default_factory=list)
    weights: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cl_usage: int = 0
    units_consumed: float = 0.0
    units_remaining: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


class VisionClient:
    """Thin synchronous client around the Vispark Vision text endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ):
        self.api_key = api_key or os.environ.get("VISION_API_KEY")
        if not self.api_key:
            raise ValueError("VISION_API_KEY not set")
        self.url = url or os.environ.get("VISION_API_URL") or DEFAULT_URL
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.Client(timeout=timeout)

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ #
    def call(
        self,
        *,
        size: VisionSize,
        content: list[dict[str, Any]],
        system_message: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        continual_learning: bool = False,
        weights: str | None = None,
    ) -> VisionResponse:
        """Make one Vision call. Returns the parsed response."""

        payload: dict[str, Any] = {
            "size": size,
            "content": content,
            "stream": False,
        }
        if system_message:
            payload["system_message"] = system_message
        if tools:
            payload["tools"] = tools
        if continual_learning:
            payload["continual_learning"] = True
            if weights:
                payload["weights"] = weights

        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self._client.post(self.url, json=payload, headers=headers)
            except httpx.HTTPError as e:
                last_exc = e
                wait = min(2**attempt, 30)
                log.warning("Vision transport error %s (attempt %d/%d) — retry in %ds",
                            e, attempt, self.max_retries, wait)
                time.sleep(wait)
                continue

            if r.status_code == 200:
                return self._parse(r.json())

            body_text = (r.text or "")[:500]
            # Insufficient tokens => training stop signal, do not retry
            if r.status_code in (400, 413) and any(
                m in body_text.lower() for m in INSUFFICIENT_TOKEN_MARKERS
            ):
                raise VisionInsufficientTokensError(
                    f"Vision API reports insufficient tokens: {body_text}"
                )

            if r.status_code in (429, 500, 502, 503, 504):
                wait = min(2**attempt, 30)
                log.warning("Vision API %d (attempt %d/%d) — retry in %ds. Body: %s",
                            r.status_code, attempt, self.max_retries, wait, body_text)
                time.sleep(wait)
                continue

            # Non-retryable
            raise VisionAPIError(f"Vision API {r.status_code}: {body_text}")

        raise VisionAPIError(f"Vision API exhausted retries: {last_exc}")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse(payload: dict[str, Any]) -> VisionResponse:
        data = payload.get("data") or {}
        kind = data.get("type") or "text"
        weights = data.get("weights")

        tool_calls: list[VisionToolCall] = []
        if kind == "tool_calls":
            for tc in data.get("tool_calls") or []:
                tool_calls.append(VisionToolCall(
                    name=str(tc.get("name") or ""),
                    arguments=tc.get("arguments") or {},
                ))

        return VisionResponse(
            type="tool_calls" if kind == "tool_calls" else "text",
            content=str(data.get("content") or ""),
            tool_calls=tool_calls,
            weights=weights,
            input_tokens=int(data.get("input_tokens") or 0),
            output_tokens=int(data.get("output_tokens") or 0),
            cl_usage=int(data.get("continual_learning_usage") or 0),
            units_consumed=float(payload.get("units_consumed") or 0.0),
            units_remaining=float(payload.get("units_remaining") or 0.0),
            raw=payload,
        )
