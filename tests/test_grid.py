"""Tests for grid rendering and diffing — no network."""

from __future__ import annotations

import numpy as np

from vision_arc_agi.grid import (
    ASCII_CHARS,
    color_histogram,
    diff_grids,
    grid_to_ascii,
    grid_to_png_bytes,
    hash_grid,
    to_numpy,
)


def test_to_numpy_clips_and_shapes():
    g = [[0, 1, 2], [3, 99, -5]]
    arr = to_numpy(g)
    assert arr.dtype == np.uint8
    assert arr.shape == (2, 3)
    # 99 clipped to 15, -5 clipped to 0
    assert arr[1, 1] == 15
    assert arr[1, 2] == 0


def test_grid_to_ascii_round_trip():
    g = [[i for i in range(16)] for _ in range(2)]
    text = grid_to_ascii(g, with_axes=False)
    # 2 rows, each 16 chars long
    rows = text.split("\n")
    assert len(rows) == 2
    for r in rows:
        assert len(r) == 16
        for ch in r:
            assert ch in ASCII_CHARS


def test_hash_grid_stable_and_changes():
    g1 = np.zeros((8, 8), dtype=np.int16)
    g2 = g1.copy()
    g2[0, 0] = 1
    h1 = hash_grid(g1)
    assert h1 == hash_grid(g1.copy())
    assert h1 != hash_grid(g2)


def test_diff_grids_summary():
    a = np.zeros((4, 4), dtype=np.int16)
    b = a.copy()
    b[1, 2] = 3
    b[3, 0] = 7
    d = diff_grids(a, b)
    assert d["changed_cells"] == 2
    assert d["bbox"] == (1, 0, 3, 2)
    assert d["by_transition"]["0->3"] == 1
    assert d["by_transition"]["0->7"] == 1


def test_diff_grids_no_change():
    a = np.zeros((3, 3), dtype=np.int16)
    d = diff_grids(a, a.copy())
    assert d["changed_cells"] == 0
    assert d["bbox"] is None


def test_color_histogram_counts():
    g = [[0, 0, 1], [2, 2, 2]]
    h = color_histogram(g)
    assert h == {0: 2, 1: 1, 2: 3}


def test_grid_to_png_bytes_valid_png():
    g = [[(i + j) % 16 for i in range(8)] for j in range(8)]
    data = grid_to_png_bytes(g, cell_size=4)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
