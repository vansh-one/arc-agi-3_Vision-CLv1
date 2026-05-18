"""Tool specs exposed to Vision + a local dispatcher.

The harness gives Vision a strict tool surface:

- ``take_action`` — submit ONE ARC action this turn (1-7, optional x/y)
- ``take_action_sequence`` — submit a list of NO-DATA actions (1-5, 7) that
  the agent is confident will succeed deterministically; lets Vision burn
  through known-good motion paths without 1 LLM call per step
- ``analyze_with_python`` — sandboxed python snippet over ``grid`` for
  computing patterns, distances, clusters; result fed back next turn
- ``reset_level`` — issue RESET (used for stuck/restart)
- ``concede_level`` — explicitly give up the current level (move on)

Internal tool calls don't count as ARC actions per the docs; we exploit that
to let Vision think with code before committing to a real action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---- Tool specs (function schema for the Vision API) --------------- #

TOOL_DEFS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "take_action",
            "description": (
                "Submit exactly ONE ARC-AGI-3 game action this turn. "
                "Action codes: 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=INTERACT/SELECT, "
                "6=CLICK (requires x,y in 0..63), 7=UNDO. Use the most decisive "
                "action you can justify from the current frame. Always include "
                "'reasoning' and a brief 'expected' description of what the "
                "frame should look like after this action so we can detect "
                "surprise."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "integer",
                        "description": "Action code 1-7",
                        "minimum": 1,
                        "maximum": 7,
                    },
                    "x": {
                        "type": "integer",
                        "description": "X coord for action 6 (0..63)",
                        "minimum": 0,
                        "maximum": 63,
                    },
                    "y": {
                        "type": "integer",
                        "description": "Y coord for action 6 (0..63)",
                        "minimum": 0,
                        "maximum": 63,
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "1-3 sentences: why this action, what hypothesis it tests.",
                    },
                    "expected": {
                        "type": "string",
                        "description": "Briefly describe the expected frame change after this action.",
                    },
                },
                "required": ["action", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "take_action_sequence",
            "description": (
                "Submit a list of 2-8 NO-DATA actions (codes 1,2,3,4,5,7) that "
                "you are highly confident will succeed in sequence (e.g. 'move "
                "right 4 times to reach the wall'). The harness executes them "
                "one at a time and STOPS early if WIN/GAME_OVER fires or if the "
                "grid stops changing. Use this only when the level mechanic is "
                "deterministic and well-understood. NEVER include action 6 in a "
                "sequence — use take_action for clicks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1, "maximum": 7},
                        "minItems": 2,
                        "maxItems": 8,
                    },
                    "reasoning": {"type": "string"},
                },
                "required": ["actions", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_with_python",
            "description": (
                "Run a short python snippet against the current grid for "
                "diagnostic computation BEFORE deciding the next action. "
                "Variable `grid` (alias `G`) is a 64x64 numpy array of ints "
                "0..15. Pre-imported: np, math, Counter, defaultdict, deque, "
                "permutations, combinations, product, plus helpers "
                "find_color(G, c), flood_clusters(G, c), bbox(mask). Use "
                "print() to surface what you want to see — output is capped "
                "at 4 KB. This call does NOT consume an ARC action."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source."},
                    "purpose": {"type": "string"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reset_level",
            "description": (
                "Issue RESET to restart the current level. Use ONLY when "
                "thoroughly stuck (8+ no-change actions, irrecoverable state, "
                "or a known dead-end). Resets cost actions and reset the "
                "level efficiency clock."
            ),
            "parameters": {
                "type": "object",
                "properties": {"reasoning": {"type": "string"}},
                "required": ["reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "concede_level",
            "description": (
                "Give up this level and move the game forward (effectively a "
                "RESET, marked as conceded). Use only when the level has "
                "consumed more than 4x the baseline action budget AND we are "
                "not making progress. The harness will record the concession "
                "and stop attempting this level for the current pass."
            ),
            "parameters": {
                "type": "object",
                "properties": {"reasoning": {"type": "string"}},
                "required": ["reasoning"],
            },
        },
    },
]


# ---- Parsed shapes ------------------------------------------------- #

@dataclass
class ToolDecision:
    """The agent's atomic decision for one turn."""

    kind: str  # 'action' | 'sequence' | 'analysis' | 'reset' | 'concede' | 'invalid'
    action: int | None = None
    x: int | None = None
    y: int | None = None
    sequence: list[int] | None = None
    code: str | None = None
    reasoning: str = ""
    expected: str = ""
    raw_name: str = ""
    raw_args: dict[str, Any] | None = None


def parse_tool_call(name: str, args: dict[str, Any]) -> ToolDecision:
    """Map a Vision tool_call to a ToolDecision."""

    args = args or {}
    common = {
        "raw_name": name,
        "raw_args": args,
        "reasoning": str(args.get("reasoning") or "")[:1500],
    }
    if name == "take_action":
        a = args.get("action")
        try:
            a = int(a)
        except (TypeError, ValueError):
            return ToolDecision(kind="invalid", **common)
        if a < 1 or a > 7:
            return ToolDecision(kind="invalid", **common)
        x = args.get("x")
        y = args.get("y")
        if a == 6:
            try:
                xi = int(x)
                yi = int(y)
            except (TypeError, ValueError):
                return ToolDecision(kind="invalid", **common)
            if not (0 <= xi <= 63 and 0 <= yi <= 63):
                return ToolDecision(kind="invalid", **common)
            return ToolDecision(
                kind="action", action=6, x=xi, y=yi,
                expected=str(args.get("expected") or "")[:400], **common,
            )
        return ToolDecision(
            kind="action", action=a,
            expected=str(args.get("expected") or "")[:400], **common,
        )

    if name == "take_action_sequence":
        seq = args.get("actions") or []
        try:
            seq = [int(a) for a in seq]
        except (TypeError, ValueError):
            return ToolDecision(kind="invalid", **common)
        seq = [a for a in seq if 1 <= a <= 7 and a != 6]
        if not (2 <= len(seq) <= 8):
            return ToolDecision(kind="invalid", **common)
        return ToolDecision(kind="sequence", sequence=seq, **common)

    if name == "analyze_with_python":
        code = args.get("code") or ""
        if not isinstance(code, str) or not code.strip():
            return ToolDecision(kind="invalid", **common)
        return ToolDecision(kind="analysis", code=code, **common)

    if name == "reset_level":
        return ToolDecision(kind="reset", **common)

    if name == "concede_level":
        return ToolDecision(kind="concede", **common)

    return ToolDecision(kind="invalid", **common)
