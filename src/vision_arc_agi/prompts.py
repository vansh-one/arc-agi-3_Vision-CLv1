"""System prompt + per-turn message builder for the Vision agent.

The system prompt is constant across all training and competition turns; the
per-turn message contains the live frame, history, warnings, and the explicit
question for the model. Continual-Learning weights carry latent skill across
calls — the per-turn message stays compact so the model spends its attention
budget on the current decision, not re-reading static instructions.
"""

from __future__ import annotations

from typing import Any

from .grid import ASCII_CHARS, color_histogram, color_legend_text, grid_to_ascii
from .state import GameState

# --------------------------------------------------------------------- #
# System prompt (constant)
# --------------------------------------------------------------------- #

SYSTEM_PROMPT = f"""You are an autonomous agent playing ARC-AGI-3. Your job is to
WIN games using as few actions as possible — your per-level score is
(human_baseline_actions / your_actions)^2, capped at ~1.32 and weighted by
level index, so later levels matter more than early ones.

== Coordinate system ==
- Grids are up to 64x64. ORIGIN (0,0) is TOP-LEFT.
- X is the COLUMN (rightwards). Y is the ROW (downwards).
- For ACTION6 you supply integer x,y in 0..63.

== Action codes ==
- ACTION1 = move UP / context-up
- ACTION2 = move DOWN
- ACTION3 = move LEFT
- ACTION4 = move RIGHT
- ACTION5 = interact / select / rotate / fire / advance
- ACTION6 = CLICK at (x,y) — only this action carries data
- ACTION7 = UNDO (if the game supports it)
- RESET   = restart current level (issued via the reset_level tool)

Not every game uses every action. The frame metadata lists
"available_actions" for the current state — restrict yourself to those.
Game-level "tags" hint at the action surface:
  - "keyboard"        → action1-5,7 only
  - "click"           → mainly action6
  - "keyboard_click"  → both
  - empty             → check available_actions each step

== Color legend (ASCII map) ==
The grid is shown to you as fixed-width characters AND as an image.
{color_legend_text()}

== Tool surface ==
You have four tools every turn. Each turn you must emit ONE tool call:

1. take_action — submit exactly one ARC action. Use this when you've made up
   your mind. Always justify with `reasoning` and an `expected` post-frame.
2. take_action_sequence — 2-8 deterministic no-data actions in a row, executed
   one-at-a-time with auto-stop on WIN / GAME_OVER / no-change. Use when you
   are confident about a fixed motion path.
3. analyze_with_python — run a sandboxed snippet against `grid` (np.uint8 64x64)
   to compute clusters, distances, symmetry, possible click targets, etc.
   This does NOT consume an ARC action — Vision's own tool calls are free per
   ARC scoring rules. Use it once or twice per level when stuck, not every turn.
4. reset_level — restart current level. Use only after stuck warning fires AND
   you can't break the deadlock another way. Resets cost actions.
5. concede_level — give up this level (4x baseline budget without progress).

== Decision policy ==
- READ the frame carefully. Identify what changed since the last turn (the
  diff summary is provided). Form a hypothesis about the mechanic.
- PREFER take_action_sequence for routing moves once a target is identified.
- For grids you do not understand yet: take ONE careful action and observe.
- Continual Learning is enabled. Your weights persist across turns, levels,
  and games — facts about a game's mechanic, palette, or layout you note now
  will survive into the competition phase. Be a careful and patient learner.

== Scoring intuition ==
- baseline=50, you take 50 → score = 1.00
- baseline=50, you take 100 → score = 0.25  ←  half the credit for 2x actions
- baseline=50, you take 75 → score ≈ 0.44
- baseline=50, you take 43 → score = 1.35 (capped at 1.32)
- Resets within a level count as actions and break the efficiency budget.
- Conceding/leaving NOT_FINISHED loses ALL credit for that level.

== Output format ==
Always respond by calling exactly ONE tool. Do not write prose outside the
tool call. The harness will reply on the next turn with the resulting frame.
"""


# --------------------------------------------------------------------- #
# Per-turn message
# --------------------------------------------------------------------- #

def render_initial_turn_text(state: GameState, *, phase: str, run_id: str, attempt: int) -> str:
    """Text rendered on the FIRST turn of a new game.

    Includes long-form metadata that we don't want to re-send each turn
    (continual-learning weights carry it).
    """
    bl_str = ", ".join(str(b) for b in state.baseline_actions) or "?"
    tags = ", ".join(state.tags) or "(none)"
    return (
        f"=== NEW GAME ({phase} phase, attempt {attempt}) ===\n"
        f"run_id      = {run_id}\n"
        f"game_id     = {state.game_id}\n"
        f"title       = {state.title}\n"
        f"tags        = {tags}\n"
        f"levels      = {len(state.baseline_actions)}\n"
        f"baselines   = [{bl_str}]   (median human actions per level)\n"
        f"win_levels  = {state.win_levels}\n"
        f"\nPlay efficiently. Form a model of the mechanic. Below is the\n"
        f"initial frame after RESET.\n"
    )


def render_turn_text(
    state: GameState,
    *,
    diff_summary: dict[str, Any] | None,
    last_step: dict[str, Any] | None,
    last_analysis: str | None,
) -> str:
    """Text part of a per-turn observation."""

    parts: list[str] = []
    parts.append(state.progress_summary())
    parts.append(f"state        = {state.current_state}")
    parts.append(
        f"available    = {state.available_actions}   "
        f"(grid hash {state.current_hash})"
    )

    # Color histogram so the model can spot timers / counters cheaply
    hist = color_histogram(state.current_grid) if state.current_grid is not None else {}
    if hist:
        nice = " ".join(
            f"{c}:{n}" for c, n in sorted(hist.items(), key=lambda kv: -kv[1])[:8]
        )
        parts.append(f"colors       = {nice}")

    # What did our last action do?
    if last_step is not None:
        a = last_step.get("action")
        nm = f"ACTION{a}"
        if a == 6:
            nm += f"({last_step.get('x')},{last_step.get('y')})"
        elif a == 0:
            nm = "RESET"
        diff = last_step.get("diff_cells") or 0
        parts.append(f"last_action  = {nm}  → diff_cells={diff}")
        if diff_summary and diff_summary.get("by_transition"):
            parts.append(f"last_transit = {diff_summary['by_transition']}")
        if diff_summary and diff_summary.get("bbox"):
            parts.append(f"last_bbox    = {diff_summary['bbox']}")

    # Sticky analysis output from prior python tool call
    if last_analysis:
        parts.append("python_out   = " + last_analysis.strip().replace("\n", "\n               "))

    # Warnings
    for w in (state.repetition_warning(), state.stuck_warning()):
        if w:
            parts.append(w)

    tried = state.already_tried_summary()
    if tried:
        parts.append(tried)

    parts.append("")
    parts.append("=== FRAME (ASCII) ===")
    parts.append(grid_to_ascii(state.current_grid))
    parts.append("=== END FRAME ===")
    parts.append("")
    parts.append("Decide your next move and call exactly ONE tool.")

    return "\n".join(parts)


def render_analysis_followup(result: dict, snippet: str) -> str:
    """Text payload returned to Vision AFTER an analyze_with_python tool call."""
    head = f"# analyze_with_python result (ok={result.get('ok')})"
    err = result.get("error") or ""
    out = result.get("stdout") or ""
    parts = [head]
    if err:
        parts.append(f"ERROR: {err}")
    if out:
        parts.append("STDOUT:")
        parts.append(out)
    parts.append("")
    parts.append("Now decide the next ARC action (take_action or take_action_sequence).")
    return "\n".join(parts)


# Exposed for unit tests
__all__ = [
    "SYSTEM_PROMPT",
    "render_initial_turn_text",
    "render_turn_text",
    "render_analysis_followup",
    "ASCII_CHARS",
]
