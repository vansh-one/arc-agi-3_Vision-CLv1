"""Tests for the python tool sandbox."""

from __future__ import annotations

import numpy as np

from vision_arc_agi.sandbox import run_python_snippet


def test_simple_print():
    res = run_python_snippet("print('hello')")
    assert res["ok"]
    assert "hello" in res["stdout"]


def test_grid_inspection():
    grid = np.array([[0, 1, 2], [3, 1, 0]], dtype=np.uint8)
    res = run_python_snippet(
        "print(int((G == 1).sum())); print('shape', G.shape)",
        grid=grid,
    )
    assert res["ok"]
    assert "2" in res["stdout"]
    assert "(2, 3)" in res["stdout"]


def test_find_color_helper():
    grid = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)
    res = run_python_snippet("print(find_color(G, 1))", grid=grid)
    assert res["ok"]
    assert "(0, 1)" in res["stdout"]
    assert "(1, 2)" in res["stdout"]


def test_error_captured():
    res = run_python_snippet("raise ValueError('boom')")
    assert not res["ok"]
    assert "ValueError" in res["error"]


def test_no_imports():
    res = run_python_snippet("import os; print(os.getcwd())")
    assert not res["ok"]
    # Either NameError (no os) or import blocked
    assert "Error" in res["error"] or "Name" in res["error"]


def test_output_capped():
    res = run_python_snippet("print('x' * 5000)")
    assert res["ok"]
    assert res["truncated"]
    assert len(res["stdout"]) < 5000 + 200
