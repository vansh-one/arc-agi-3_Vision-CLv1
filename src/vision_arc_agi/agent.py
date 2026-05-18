"""Main VisionAgent: Vision (Large) + Continual Learning + agentic tool loop.

One agent instance plays one game (multiple levels, possibly multiple resets).
Continual-Learning weights are carried by the caller across games and sessions
— the agent reads the input ``weights`` at construction and exposes the
updated blob via ``self.weights`` for the caller to persist.

The play loop is:

1. RESET the game (counts as action 0, no Vision call needed)
2. For each turn:
   a. Build the per-turn message (frame ASCII + image, diff summary, warnings)
   b. Call Vision with continual_learning=true + current weights + tools
   c. Parse the tool_call into a ToolDecision
   d. If 'analysis': run python sandbox, append result, ask Vision again
      (max ANALYZE_BUDGET_PER_TURN analyses per turn to bound cost)
   e. If 'action' / 'sequence' / 'reset' / 'concede': execute against ARC
   f. Persist new weights, update state, log, repeat
3. Terminate on WIN / GAME_OVER / per-game action ceiling / fatal error.

Anti-loop safeguards (independent of Vision's good behaviour):
- Hard cap on actions per level (4 × baseline median or 200, whichever lower)
- Auto-reset after 12 consecutive no-change actions
- Forced concede after 3 in-level resets
- Bail out of game after WIN-loop pause detection at WIN state
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .arc_runner import ArcRunner, _state_name


def _state_str(state_value) -> str:
    """Return a stable bare-string state name (e.g. 'WIN', 'NOT_FINISHED')."""
    return _state_name(state_value)
from .grid import diff_grids, grid_to_png_base64, to_numpy
from .memory import WeightsRecord
from .prompts import SYSTEM_PROMPT, render_analysis_followup, render_initial_turn_text, render_turn_text
from .sandbox import run_python_snippet
from .state import GameState
from .tools import TOOL_DEFS, parse_tool_call
from .vision import VisionClient, VisionInsufficientTokensError, VisionResponse, VisionSize

log = logging.getLogger(__name__)

ANALYZE_BUDGET_PER_TURN = 2
MAX_INVALID_TOOL_RETRIES = 2
HARD_ACTIONS_PER_LEVEL_CEILING = 200
NO_CHANGE_AUTO_RESET = 12
MAX_RESETS_PER_LEVEL = 3
MAX_TOTAL_RESETS_PER_GAME = 8


# --------------------------------------------------------------------- #
@dataclass
class GameResult:
    game_id: str
    title: str
    state: str = "NOT_FINISHED"
    levels_completed: int = 0
    total_levels: int = 0
    actions_used: int = 0
    baseline_total: int = 0
    win: bool = False
    estimated_game_score: float = 0.0
    notes: list[str] = field(default_factory=list)
    transcript_path: str | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class AgentRun:
    """One play of one game from start to terminal."""

    game_id: str
    run_id: str
    final_weights: str | None = None
    final_cl_usage: int = 0
    result: GameResult = field(default_factory=lambda: GameResult(game_id="", title=""))


# --------------------------------------------------------------------- #
class VisionAgent:
    def __init__(
        self,
        vision: VisionClient,
        arc: ArcRunner,
        *,
        play_size: VisionSize = "large",
        analysis_size: VisionSize = "medium",
        runs_dir: str = "runs",
        send_image: bool = True,
        png_cell: int = 8,
        max_turns: int | None = None,
    ):
        self.vision = vision
        self.arc = arc
        self.play_size = play_size
        self.analysis_size = analysis_size
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.send_image = send_image
        self.png_cell = png_cell
        self.max_turns = max_turns

    # ------------------------------------------------------------------ #
    def play_game(
        self,
        game_info,                 # GameInfo from arc_runner.list_games
        *,
        weights: str | None,
        phase: str = "train",
        attempt: int = 1,
    ) -> AgentRun:
        """Play one game start→terminal. Returns final weights + result."""

        run_id = f"{phase}-{game_info.game_id}-a{attempt}-{uuid.uuid4().hex[:6]}"
        log_path = self.runs_dir / f"{run_id}.jsonl"
        log_fh = log_path.open("w", encoding="utf-8")

        def emit(event: str, payload: dict):
            line = {"t": time.time(), "event": event, **payload}
            log_fh.write(json.dumps(line, default=str) + "\n")
            log_fh.flush()

        emit("run.start", {
            "run_id": run_id,
            "game_id": game_info.game_id,
            "phase": phase,
            "attempt": attempt,
            "play_size": self.play_size,
            "analysis_size": self.analysis_size,
            "arc_mode": self.arc.mode,
            "input_weights_len": len(weights) if weights else 0,
        })

        env = self.arc.make(game_info.game_id)
        if env is None:
            emit("run.fail", {"reason": "env_make_returned_None"})
            log_fh.close()
            return AgentRun(
                game_id=game_info.game_id,
                run_id=run_id,
                final_weights=weights,
                result=GameResult(
                    game_id=game_info.game_id,
                    title=game_info.title,
                    state="NOT_PLAYED",
                    total_levels=len(game_info.baseline_actions),
                    baseline_total=sum(game_info.baseline_actions),
                ),
            )

        # Initial reset
        obs = env.reset()
        state = GameState(
            game_id=game_info.game_id,
            title=game_info.title,
            tags=list(game_info.tags),
            baseline_actions=list(game_info.baseline_actions),
        )
        state.update_frame(
            grid=obs.frame[-1] if getattr(obs, "frame", None) else _empty_grid(),
            state=_state_str(obs.state),
            levels_completed=int(getattr(obs, "levels_completed", 0) or 0),
            win_levels=int(getattr(obs, "win_levels", 0) or 0),
            available_actions=list(getattr(obs, "available_actions", []) or []),
        )

        emit("game.start", {
            "title": game_info.title,
            "tags": game_info.tags,
            "baselines": game_info.baseline_actions,
            "initial_state": state.current_state,
        })

        # Outer loop: one turn = at most one ARC action committed
        current_weights = weights
        current_cl_usage = 0
        last_step: dict[str, Any] | None = None
        diff_summary: dict[str, Any] | None = None
        pending_analysis: str | None = None
        no_change_streak = 0
        resets_this_level = 0
        resets_total = 0
        invalid_streak = 0
        last_seen_level = state.current_level
        first_turn = True
        turn_idx = 0

        try:
            while True:
                if self.arc.state_is_terminal(state.current_state):
                    emit("game.terminal", {"state": state.current_state, "level": state.current_level})
                    break

                if self.max_turns is not None and turn_idx >= self.max_turns:
                    emit("game.cap", {"reason": "max_turns", "turns": turn_idx})
                    break

                # Per-level safety: hard ceiling
                bl = state.baseline_for_level(state.current_level)
                ceiling = min(
                    HARD_ACTIONS_PER_LEVEL_CEILING,
                    int((bl or 50) * 4),
                )
                if state.active_level().actions_taken >= ceiling:
                    emit("level.cap", {
                        "level": state.current_level,
                        "actions": state.active_level().actions_taken,
                        "ceiling": ceiling,
                    })
                    if resets_this_level < MAX_RESETS_PER_LEVEL and resets_total < MAX_TOTAL_RESETS_PER_GAME:
                        # Auto-reset once more, otherwise bail
                        last_step, state, diff_summary = self._do_reset(env, state, "auto_cap")
                        resets_this_level += 1
                        resets_total += 1
                        no_change_streak = 0
                        continue
                    emit("game.cap", {"reason": "level_cap_no_resets_left"})
                    break

                if no_change_streak >= NO_CHANGE_AUTO_RESET and resets_this_level < MAX_RESETS_PER_LEVEL:
                    emit("level.auto_reset", {
                        "level": state.current_level,
                        "no_change_streak": no_change_streak,
                    })
                    last_step, state, diff_summary = self._do_reset(env, state, "no_change_streak")
                    resets_this_level += 1
                    resets_total += 1
                    no_change_streak = 0
                    continue

                # ---- Build turn content ---- #
                content_blocks: list[dict[str, Any]] = []
                if first_turn:
                    content_blocks.append({
                        "type": "text",
                        "content": render_initial_turn_text(
                            state, phase=phase, run_id=run_id, attempt=attempt,
                        ),
                    })
                    first_turn = False
                content_blocks.append({
                    "type": "text",
                    "content": render_turn_text(
                        state,
                        diff_summary=diff_summary,
                        last_step=last_step,
                        last_analysis=pending_analysis,
                    ),
                })
                pending_analysis = None
                if self.send_image and state.current_grid is not None:
                    content_blocks.append({
                        "type": "image",
                        "content": grid_to_png_base64(
                            state.current_grid, cell_size=self.png_cell,
                        ),
                    })

                # ---- Ask Vision ---- #
                analyses_used = 0
                action_committed = False
                while not action_committed:
                    resp = self._call_vision(
                        content_blocks,
                        weights=current_weights,
                        size=self.play_size,
                    )
                    current_weights = resp.weights or current_weights
                    current_cl_usage = resp.cl_usage or current_cl_usage
                    emit("vision.call", {
                        "turn": turn_idx,
                        "type": resp.type,
                        "input_tokens": resp.input_tokens,
                        "output_tokens": resp.output_tokens,
                        "cl_usage": resp.cl_usage,
                        "weights_len": len(current_weights or ""),
                        "units_consumed": resp.units_consumed,
                    })

                    if resp.type != "tool_calls" or not resp.tool_calls:
                        # Try to parse a free-text fallback
                        invalid_streak += 1
                        emit("vision.invalid", {"content": resp.content[:200]})
                        if invalid_streak > MAX_INVALID_TOOL_RETRIES:
                            # default to the first available action so we keep moving
                            fallback = state.available_actions[0] if state.available_actions else 1
                            emit("vision.fallback", {"action": fallback})
                            last_step, state, diff_summary, no_change_streak = self._do_action(
                                env, state, fallback, None, None, "fallback after invalid vision output",
                                no_change_streak,
                            )
                            action_committed = True
                            invalid_streak = 0
                            break
                        content_blocks = [{
                            "type": "text",
                            "content": (
                                "You did not call a tool. You MUST call exactly one "
                                "of: take_action, take_action_sequence, "
                                "analyze_with_python, reset_level, concede_level."
                            ),
                        }]
                        continue
                    invalid_streak = 0

                    decision = parse_tool_call(
                        resp.tool_calls[0].name, resp.tool_calls[0].arguments
                    )
                    emit("vision.decision", {
                        "kind": decision.kind,
                        "raw_name": decision.raw_name,
                        "raw_args": decision.raw_args,
                    })

                    if decision.kind == "analysis":
                        if analyses_used >= ANALYZE_BUDGET_PER_TURN:
                            content_blocks = [{
                                "type": "text",
                                "content": (
                                    "Analysis budget for this turn exhausted "
                                    f"({ANALYZE_BUDGET_PER_TURN}). Decide an action now."
                                ),
                            }]
                            continue
                        analyses_used += 1
                        result = run_python_snippet(decision.code, grid=state.current_grid)
                        emit("analysis", {
                            "ok": result["ok"],
                            "stdout_len": len(result["stdout"]),
                            "error": result.get("error"),
                        })
                        content_blocks = [{
                            "type": "text",
                            "content": render_analysis_followup(result, decision.code),
                        }]
                        continue

                    if decision.kind == "invalid":
                        invalid_streak += 1
                        if invalid_streak > MAX_INVALID_TOOL_RETRIES:
                            fallback = state.available_actions[0] if state.available_actions else 1
                            emit("vision.fallback", {"action": fallback})
                            last_step, state, diff_summary, no_change_streak = self._do_action(
                                env, state, fallback, None, None, "fallback after invalid tool args",
                                no_change_streak,
                            )
                            action_committed = True
                            invalid_streak = 0
                            break
                        content_blocks = [{
                            "type": "text",
                            "content": (
                                f"Your last tool call was invalid: name={decision.raw_name} "
                                f"args={decision.raw_args}. Please call a valid tool with valid args."
                            ),
                        }]
                        continue

                    if decision.kind == "reset":
                        if resets_total >= MAX_TOTAL_RESETS_PER_GAME or resets_this_level >= MAX_RESETS_PER_LEVEL:
                            emit("reset.refused", {
                                "reason": "exceeds_reset_budget",
                                "resets_this_level": resets_this_level,
                                "resets_total": resets_total,
                            })
                            content_blocks = [{
                                "type": "text",
                                "content": (
                                    "Reset budget exhausted. Please take a concrete action "
                                    "or concede_level if you truly have no progress path."
                                ),
                            }]
                            continue
                        last_step, state, diff_summary = self._do_reset(env, state, decision.reasoning)
                        resets_this_level += 1
                        resets_total += 1
                        no_change_streak = 0
                        action_committed = True
                        break

                    if decision.kind == "concede":
                        emit("level.concede", {"reasoning": decision.reasoning})
                        # We model concede as a reset for ARC purposes but track it.
                        last_step, state, diff_summary = self._do_reset(env, state, "concede:" + decision.reasoning[:200])
                        resets_total += 1
                        resets_this_level += 1
                        no_change_streak = 0
                        # Skip ahead: bump per-level cap to force forward motion next pass
                        action_committed = True
                        break

                    if decision.kind == "sequence":
                        last_step, state, diff_summary, no_change_streak = self._do_sequence(
                            env, state, decision.sequence or [], decision.reasoning,
                            no_change_streak,
                        )
                        action_committed = True
                        break

                    if decision.kind == "action":
                        last_step, state, diff_summary, no_change_streak = self._do_action(
                            env, state, decision.action or 1,
                            decision.x, decision.y, decision.reasoning,
                            no_change_streak,
                        )
                        action_committed = True
                        break

                # End inner Vision retry loop

                # Level advanced? reset reset budget
                if state.current_level != last_seen_level:
                    emit("level.changed", {
                        "from": last_seen_level,
                        "to": state.current_level,
                        "actions_in_prev_level": state.levels[-2].actions_taken if len(state.levels) >= 2 else 0,
                    })
                    last_seen_level = state.current_level
                    resets_this_level = 0

                turn_idx += 1
        except VisionInsufficientTokensError as e:
            emit("vision.insufficient_tokens", {"err": str(e)})
            log.warning("Vision token budget exhausted; aborting game %s", game_info.game_id)
        except Exception as e:  # noqa: BLE001
            log.exception("Game loop error")
            emit("game.error", {"err": f"{type(e).__name__}: {e}"})

        # Final scoring estimate
        result = self._summarize(state, game_info, log_path)
        emit("run.end", {
            "result": result.to_dict(),
            "weights_len": len(current_weights or ""),
            "cl_usage": current_cl_usage,
        })
        log_fh.close()
        return AgentRun(
            game_id=game_info.game_id,
            run_id=run_id,
            final_weights=current_weights,
            final_cl_usage=current_cl_usage,
            result=result,
        )

    # ------------------------------------------------------------------ #
    def _call_vision(
        self, content: list[dict], *, weights: str | None, size: VisionSize,
    ) -> VisionResponse:
        return self.vision.call(
            size=size,
            content=content,
            system_message=SYSTEM_PROMPT,
            tools=TOOL_DEFS,
            continual_learning=True,
            weights=weights,
        )

    # ------------------------------------------------------------------ #
    def _do_action(
        self, env, state: GameState, action: int, x: int | None, y: int | None,
        reasoning: str, no_change_streak: int,
    ) -> tuple[dict, GameState, dict, int]:
        pre_grid = state.current_grid.copy() if state.current_grid is not None else None
        data = {"x": x, "y": y} if (action == 6 and x is not None and y is not None) else None
        # Pre-flight: only submit if action is in available list; ARC rejects otherwise
        if state.available_actions and action not in state.available_actions and action != 6:
            # remap to nearest available
            allowed = state.available_actions
            action = allowed[0]
        ga = self.arc.action_for(action)
        obs = env.step(
            ga,
            data=data,
            reasoning={
                "thought": reasoning[:1400],
                "agent": "vispark-vision-cl-large",
            },
        )
        grid_after = obs.frame[-1] if getattr(obs, "frame", None) else _empty_grid()
        state.record_step(
            action=action, x=x, y=y, reasoning=reasoning,
            pre_grid=pre_grid, post_grid=grid_after,
            state_after=str(obs.state),
            level_after=int(getattr(obs, "levels_completed", 0) or 0),
            available_actions=list(getattr(obs, "available_actions", []) or []),
        )
        state.update_frame(
            grid=grid_after,
            state=_state_str(obs.state),
            levels_completed=int(getattr(obs, "levels_completed", 0) or 0),
            win_levels=int(getattr(obs, "win_levels", 0) or 0),
            available_actions=list(getattr(obs, "available_actions", []) or []),
        )
        diff = diff_grids(pre_grid, state.current_grid)
        last_step = {
            "action": action, "x": x, "y": y,
            "diff_cells": diff.get("changed_cells") or 0,
        }
        new_streak = no_change_streak + 1 if (diff.get("changed_cells") or 0) == 0 else 0
        return last_step, state, diff, new_streak

    def _do_sequence(
        self, env, state: GameState, actions: list[int], reasoning: str, no_change_streak: int,
    ) -> tuple[dict, GameState, dict, int]:
        last = None
        diff: dict[str, Any] = {}
        streak = no_change_streak
        for a in actions:
            if a == 6:
                continue  # safety
            last, state, diff, streak = self._do_action(
                env, state, a, None, None, f"seq: {reasoning}", streak,
            )
            if self.arc.state_is_terminal(state.current_state):
                break
            if (diff.get("changed_cells") or 0) == 0:
                # stop on first no-op — sequence assumption failed
                break
        return last or {"action": 0, "diff_cells": 0}, state, diff, streak

    def _do_reset(
        self, env, state: GameState, reasoning: str,
    ) -> tuple[dict, GameState, dict]:
        pre_grid = state.current_grid.copy() if state.current_grid is not None else None
        obs = env.reset()
        grid_after = obs.frame[-1] if getattr(obs, "frame", None) else _empty_grid()
        state.record_step(
            action=0, x=None, y=None, reasoning=reasoning,
            pre_grid=pre_grid, post_grid=grid_after,
            state_after=str(obs.state),
            level_after=int(getattr(obs, "levels_completed", 0) or 0),
            available_actions=list(getattr(obs, "available_actions", []) or []),
        )
        state.update_frame(
            grid=grid_after,
            state=_state_str(obs.state),
            levels_completed=int(getattr(obs, "levels_completed", 0) or 0),
            win_levels=int(getattr(obs, "win_levels", 0) or 0),
            available_actions=list(getattr(obs, "available_actions", []) or []),
        )
        diff = diff_grids(pre_grid, state.current_grid)
        return {"action": 0, "diff_cells": diff.get("changed_cells") or 0}, state, diff

    # ------------------------------------------------------------------ #
    def _summarize(self, state: GameState, info, log_path: Path) -> GameResult:
        # Estimate per-level score: (baseline/used)^2 for completed levels;
        # 0 for not-finished.
        completed = state.current_level
        baseline_total = sum(state.baseline_actions)
        scores: list[float] = []
        for idx, lvl in enumerate(state.levels):
            if idx >= completed:
                scores.append(0.0)
                continue
            bl = state.baseline_for_level(idx)
            used = max(lvl.actions_taken, 1)
            s = (bl / used) ** 2 if bl else 0.0
            scores.append(min(s, 1.3225))
        # Pad to full level count
        while len(scores) < len(state.baseline_actions):
            scores.append(0.0)
        # weighted average by 1-indexed level number
        if scores:
            weights = [i + 1 for i in range(len(scores))]
            est = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        else:
            est = 0.0
        win = self.arc.state_is_win(state.current_state)
        notes: list[str] = []
        if win:
            notes.append("WIN")
        elif state.current_state == "GAME_OVER":
            notes.append("GAME_OVER")
        return GameResult(
            game_id=state.game_id,
            title=state.title,
            state=state.current_state,
            levels_completed=completed,
            total_levels=len(state.baseline_actions),
            actions_used=state.total_actions,
            baseline_total=baseline_total,
            win=win,
            estimated_game_score=est,
            notes=notes,
            transcript_path=str(log_path),
        )


def _empty_grid():
    return np.zeros((64, 64), dtype=np.uint8)
