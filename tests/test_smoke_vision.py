"""Live smoke test against the real Vision API.

Skipped unless ``VISION_API_KEY`` is set, so unit-test CI can run without it.
Hits the Vision Small endpoint with a simulated turn payload and checks that
the response is a parseable tool call.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from vision_arc_agi.grid import grid_to_png_base64
from vision_arc_agi.prompts import SYSTEM_PROMPT, render_initial_turn_text, render_turn_text
from vision_arc_agi.state import GameState
from vision_arc_agi.tools import TOOL_DEFS, parse_tool_call
from vision_arc_agi.vision import VisionClient


pytestmark = pytest.mark.skipif(
    not os.environ.get("VISION_API_KEY"),
    reason="VISION_API_KEY not set — skipping live Vision smoke",
)


def _toy_grid():
    g = np.zeros((16, 16), dtype=np.uint8)
    # red dot at top-left, blue square at bottom-right; target shape pattern
    g[0, 0] = 2
    g[12:15, 12:15] = 1
    return g


def test_live_turn_returns_tool_call():
    state = GameState(
        game_id="dev-toy",
        title="DEV",
        tags=["click"],
        baseline_actions=[30],
    )
    state.update_frame(_toy_grid(), "NOT_FINISHED", 0, 1, [1, 2, 3, 4, 5, 6])
    content = [
        {"type": "text",
         "content": render_initial_turn_text(state, phase="smoke", run_id="t1", attempt=1)},
        {"type": "text",
         "content": render_turn_text(state, diff_summary=None, last_step=None, last_analysis=None)},
        {"type": "image", "content": grid_to_png_base64(state.current_grid, cell_size=8)},
    ]
    with VisionClient() as v:
        resp = v.call(
            size="small",  # cheap for smoke
            content=content,
            system_message=SYSTEM_PROMPT,
            tools=TOOL_DEFS,
            continual_learning=True,
        )
    assert resp.type == "tool_calls", f"Vision did not return a tool call: {resp.content!r}"
    assert resp.tool_calls, "tool_calls empty"
    decision = parse_tool_call(resp.tool_calls[0].name, resp.tool_calls[0].arguments)
    assert decision.kind in ("action", "sequence", "analysis", "reset", "concede", "invalid")
    assert resp.weights, "expected continual-learning weights blob in response"
