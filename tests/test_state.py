"""Tests for game state tracking."""

from __future__ import annotations

import numpy as np

from vision_arc_agi.state import GameState


def _grid(seed: int = 0):
    g = np.zeros((4, 4), dtype=np.uint8)
    if seed:
        g[0, 0] = seed
    return g


def test_start_game_and_levels():
    s = GameState(game_id="x", title="X", baseline_actions=[10, 20, 30])
    s.update_frame(_grid(), "NOT_FINISHED", 0, 254, [1, 2])
    assert s.current_level == 0
    assert len(s.levels) == 1
    assert s.baseline_for_level(0) == 10


def test_record_step_and_actions_taken_counts_resets():
    s = GameState(game_id="x", baseline_actions=[10])
    a = _grid(0)
    s.update_frame(a, "NOT_FINISHED", 0, 1, [1, 2])
    b = _grid(1)
    s.record_step(action=1, x=None, y=None, reasoning="up",
                  pre_grid=a, post_grid=b,
                  state_after="NOT_FINISHED", level_after=0,
                  available_actions=[1, 2])
    s.record_step(action=0, x=None, y=None, reasoning="reset",
                  pre_grid=b, post_grid=a,
                  state_after="NOT_FINISHED", level_after=0,
                  available_actions=[1, 2])
    assert s.active_level().actions_taken == 2  # includes reset
    assert len(s.active_level().steps) == 2


def test_repetition_warning():
    s = GameState(game_id="x", baseline_actions=[10])
    a = _grid(0)
    s.update_frame(a, "NOT_FINISHED", 0, 1, [1])
    for _ in range(3):
        s.record_step(action=1, x=None, y=None, reasoning="up",
                      pre_grid=a, post_grid=a.copy(),
                      state_after="NOT_FINISHED", level_after=0,
                      available_actions=[1])
    assert "no change" in s.repetition_warning()


def test_baseline_score_estimate_in_summary():
    s = GameState(game_id="x", baseline_actions=[20])
    s.update_frame(_grid(), "NOT_FINISHED", 0, 1, [1])
    text = s.progress_summary()
    assert "baseline=20" in text
