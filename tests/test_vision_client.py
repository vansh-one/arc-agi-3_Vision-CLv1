"""Vision client tests — uses a fake transport, no network."""

from __future__ import annotations

import json
import re

import httpx

from vision_arc_agi.vision import (
    VisionAPIError,
    VisionClient,
    VisionInsufficientTokensError,
    VisionResponse,
)


def _client_with_handler(handler):
    transport = httpx.MockTransport(handler)
    c = VisionClient(api_key="vl_fake", url="https://test/model/text/vision", max_retries=2)
    c._client.close()
    c._client = httpx.Client(transport=transport, timeout=5)
    return c


def test_parses_text_response():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "status": "success",
            "data": {
                "type": "text",
                "content": "hi there",
                "weights": "wbase64",
                "input_tokens": 10,
                "output_tokens": 2,
                "continual_learning_usage": 50,
            },
            "units_consumed": 0.1,
            "units_remaining": 999.9,
        })
    c = _client_with_handler(handler)
    r = c.call(size="small", content=[{"type": "text", "content": "hello"}],
               continual_learning=True, weights="prev")
    assert isinstance(r, VisionResponse)
    assert r.type == "text"
    assert r.content == "hi there"
    assert r.weights == "wbase64"
    assert r.cl_usage == 50
    assert r.units_remaining == 999.9


def test_parses_tool_call_response():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "data": {
                "type": "tool_calls",
                "content": None,
                "tool_calls": [
                    {"name": "take_action", "arguments": {"action": 3, "reasoning": "left"}}
                ],
                "weights": "wb",
                "input_tokens": 10,
                "output_tokens": 5,
                "continual_learning_usage": 12,
            },
        })
    c = _client_with_handler(handler)
    r = c.call(size="medium", content=[{"type": "text", "content": "x"}], tools=[])
    assert r.type == "tool_calls"
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].name == "take_action"
    assert r.tool_calls[0].arguments["action"] == 3


def test_insufficient_tokens_raises():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "insufficient tokens, context limit exceeded"})
    c = _client_with_handler(handler)
    try:
        c.call(size="large", content=[{"type": "text", "content": "x"}])
    except VisionInsufficientTokensError:
        return
    raise AssertionError("expected VisionInsufficientTokensError")


def test_500_retries_then_fails():
    calls = {"n": 0}
    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, text="bad")
    c = _client_with_handler(handler)
    # Make backoff short for the test
    import vision_arc_agi.vision as v
    orig = v.time.sleep
    v.time.sleep = lambda *_: None
    try:
        try:
            c.call(size="small", content=[{"type": "text", "content": "x"}])
        except VisionAPIError:
            pass
    finally:
        v.time.sleep = orig
    assert calls["n"] >= 2
