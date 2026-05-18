"""Per-level game state tracking: history, repetition memo, stuck detection.

Continual-learning weights carry most of Vision's memory, but a deterministic
state-action memo and a recency window give the agent crucial guardrails:
no-change repetition warnings, action-budget bookkeeping, and a per-grid
"already tried" table that lets the model avoid loops without burning weights.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .grid import diff_grids, hash_grid, to_numpy


@dataclass
class Step:
    action: int                       # 1-7 or 0 (RESET)
    x: int | None = None
    y: int | None = None
    reasoning: str = ""
    pre_hash: str = ""
    post_hash: str = ""
    diff_cells: int = 0
    state_after: str = "NOT_FINISHED"
    level_after: int = 0
    available_actions: list[int] = field(default_factory=list)


@dataclass
class LevelHistory:
    """Tracks one level (between two level-completions / resets)."""

    level_index: int = 0
    steps: list[Step] = field(default_factory=list)
    # state_action memo: grid_hash -> {(action, x, y) -> outcome_hash}
    tried: dict[str, dict[tuple[int, int | None, int | None], str]] = field(
        default_factory=lambda: defaultdict(dict)
    )

    def record(self, step: Step):
        self.steps.append(step)
        self.tried[step.pre_hash][(step.action, step.x, step.y)] = step.post_hash

    @property
    def actions_taken(self) -> int:
        # ARC counts RESET as an action ("affects the game state"), so include it.
        return len(self.steps)

    def tried_from(self, grid_hash: str) -> list[tuple[int, int | None, int | None]]:
        return list(self.tried.get(grid_hash, {}).keys())

    def last_n_diff_cells(self, n: int = 8) -> list[int]:
        return [s.diff_cells for s in self.steps[-n:]]


@dataclass
class GameState:
    """Stateful tracker for one game (multiple levels, multiple resets)."""

    game_id: str
    title: str = ""
    tags: list[str] = field(default_factory=list)
    baseline_actions: list[int] = field(default_factory=list)

    current_grid: np.ndarray | None = None
    current_hash: str = ""
    current_state: str = "NOT_PLAYED"
    current_level: int = 0
    win_levels: int = 0
    available_actions: list[int] = field(default_factory=list)
    levels: list[LevelHistory] = field(default_factory=list)

    @property
    def total_actions(self) -> int:
        return sum(lvl.actions_taken for lvl in self.levels)

    @property
    def levels_completed(self) -> int:
        # Number of levels we have *moved past*, i.e. current_level value.
        return self.current_level

    def start_level(self, idx: int):
        if not self.levels or self.levels[-1].level_index != idx:
            self.levels.append(LevelHistory(level_index=idx))

    def active_level(self) -> LevelHistory:
        if not self.levels:
            self.start_level(0)
        return self.levels[-1]

    def update_frame(
        self,
        grid: Sequence[Sequence[int]] | np.ndarray,
        state: str,
        levels_completed: int,
        win_levels: int,
        available_actions: list[int],
    ):
        arr = to_numpy(grid)
        self.current_grid = arr
        self.current_hash = hash_grid(arr)
        self.current_state = state
        self.current_level = int(levels_completed)
        self.win_levels = int(win_levels)
        self.available_actions = list(available_actions or [])
        self.start_level(self.current_level)

    def record_step(self, action: int, x: int | None, y: int | None, reasoning: str,
                    pre_grid: np.ndarray | None, post_grid: np.ndarray,
                    state_after: str, level_after: int, available_actions: list[int]):
        pre_hash = hash_grid(pre_grid) if pre_grid is not None else ""
        post_hash = hash_grid(post_grid)
        d = diff_grids(pre_grid, post_grid)
        step = Step(
            action=action, x=x, y=y, reasoning=reasoning,
            pre_hash=pre_hash, post_hash=post_hash,
            diff_cells=int(d.get("changed_cells") or 0),
            state_after=state_after, level_after=level_after,
            available_actions=list(available_actions or []),
        )
        self.active_level().record(step)

    # --- warnings / heuristics ------------------------------------------ #

    def repetition_warning(self) -> str:
        lvl = self.active_level()
        recent = [(s.action, s.x, s.y, s.diff_cells) for s in lvl.steps[-5:]]
        if len(recent) >= 3:
            keys = [(a, x, y) for (a, x, y, _) in recent]
            if all(k == keys[0] for k in keys[-3:]) and all(d == 0 for *_, d in recent[-3:]):
                return f"⚠️  Last 3 actions {keys[0]} produced no change — consider a different move."
        return ""

    def stuck_warning(self, threshold: float = 0.7, window: int = 10) -> str:
        lvl = self.active_level()
        diffs = lvl.last_n_diff_cells(window)
        if len(diffs) < window:
            return ""
        zeros = sum(1 for d in diffs if d == 0)
        if zeros / len(diffs) >= threshold:
            return (
                f"⚠️  {zeros}/{len(diffs)} recent actions did nothing — try reset_level"
                " or fundamentally different actions."
            )
        return ""

    def already_tried_summary(self) -> str:
        """A compact list of (action, x, y) -> outcome already explored from current grid."""
        lvl = self.active_level()
        items = lvl.tried.get(self.current_hash) or {}
        if not items:
            return ""
        parts = []
        for (a, x, y), outcome in list(items.items())[:8]:
            tag = "→same" if outcome == self.current_hash else f"→{outcome[:6]}"
            who = f"ACTION{a}" + (f"({x},{y})" if a == 6 else "")
            parts.append(f"{who}{tag}")
        return "Already tried from this exact grid: " + ", ".join(parts)

    # --- summary ----------------------------------------------------- #

    def baseline_for_level(self, idx: int) -> int | None:
        if 0 <= idx < len(self.baseline_actions):
            return int(self.baseline_actions[idx])
        return None

    def progress_summary(self) -> str:
        bl = self.baseline_for_level(self.current_level)
        taken = self.active_level().actions_taken
        lvl_total = len(self.baseline_actions) or self.win_levels
        s = (
            f"game={self.game_id} title={self.title} tags={','.join(self.tags) or '-'}"
            f" | level {self.current_level + 1}/{lvl_total}"
            f" | actions_in_level={taken}"
        )
        if bl is not None:
            ratio = taken / bl if bl else 0
            score_proj = (bl / max(taken, 1)) ** 2 if taken else 1.0
            s += f" | baseline={bl} ratio={ratio:.2f}x est_lvl_score={min(score_proj, 1.3225):.2f}"
        return s
