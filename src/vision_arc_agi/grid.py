"""Render ARC-AGI-3 grids as ASCII (token-cheap) and PNG (multimodal).

The toolkit returns ``FrameDataRaw.frame`` as ``list[ndarray]`` — typically one
64x64 grid of integers 0..15. The agent prompt benefits from both modalities:

- ASCII gives Vision an exact, lossless, hashable representation of every cell.
- PNG gives Vision spatial intuition (clusters, lines, symmetry) at almost
  zero extra context cost (images are folded into the model's vision tower).
"""

from __future__ import annotations

import base64
import hashlib
import io
from collections import Counter
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Canonical ARC palette (matches the official renderer).
PALETTE: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),        # 0 black
    (30, 147, 255),   # 1 blue
    (249, 60, 49),    # 2 red
    (79, 204, 48),    # 3 green
    (255, 219, 1),    # 4 yellow
    (153, 153, 153),  # 5 gray
    (229, 58, 163),   # 6 magenta
    (255, 133, 28),   # 7 orange
    (124, 222, 236),  # 8 sky
    (146, 18, 49),    # 9 maroon
    (255, 255, 255),  # 10 white
    (180, 116, 0),    # 11 brown
    (60, 60, 60),     # 12 dark gray
    (40, 40, 40),     # 13 darker
    (20, 20, 20),     # 14 darkest
    (130, 30, 130),   # 15 purple
)

# Single-char codes for ASCII rendering. Chosen so each color has visual weight
# proportional to brightness (helps Vision parse "darker = background").
ASCII_CHARS: tuple[str, ...] = (
    ".",  # 0 black (background-like)
    "B",  # 1 blue
    "R",  # 2 red
    "G",  # 3 green
    "Y",  # 4 yellow
    "+",  # 5 gray
    "M",  # 6 magenta
    "O",  # 7 orange
    "S",  # 8 sky
    "N",  # 9 maroon
    "W",  # 10 white
    "U",  # 11 brown
    "-",  # 12 dgray
    "_",  # 13 ddgray
    ",",  # 14 dddgray
    "P",  # 15 purple
)


def to_numpy(grid: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
    """Coerce a frame to a 2-D uint8 ndarray."""
    arr = np.asarray(grid, dtype=np.int16)
    if arr.ndim == 3:
        # Some frames come as [frame_idx][row][col] — take the first.
        arr = arr[0]
    arr = np.clip(arr, 0, 15).astype(np.uint8)
    return arr


def hash_grid(grid: Sequence[Sequence[int]] | np.ndarray) -> str:
    arr = to_numpy(grid)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


def color_histogram(grid: Sequence[Sequence[int]] | np.ndarray) -> dict[int, int]:
    arr = to_numpy(grid)
    return dict(Counter(arr.flatten().tolist()))


def grid_to_ascii(grid: Sequence[Sequence[int]] | np.ndarray, *, with_axes: bool = True) -> str:
    """Render the grid as fixed-width single-char-per-cell ASCII.

    With axes (default), column header every 10 columns + row index column.
    """
    arr = to_numpy(grid)
    rows, cols = arr.shape
    lines: list[str] = []
    if with_axes:
        # Two-row header so two-digit indices fit:
        tens_row = "    " + "".join(
            (str(c // 10) if c % 10 == 0 else " ") for c in range(cols)
        )
        ones_row = "    " + "".join(str(c % 10) for c in range(cols))
        lines.append(tens_row)
        lines.append(ones_row)
    for r in range(rows):
        chars = "".join(ASCII_CHARS[int(v)] for v in arr[r])
        lines.append(f"{r:>2}: {chars}" if with_axes else chars)
    return "\n".join(lines)


def grid_to_png_bytes(
    grid: Sequence[Sequence[int]] | np.ndarray,
    *,
    cell_size: int = 12,
    with_axes: bool = True,
    with_legend: bool = True,
) -> bytes:
    """Render the grid as a PNG image with optional axes and a color legend.

    Returns raw PNG bytes.
    """
    arr = to_numpy(grid)
    rows, cols = arr.shape

    margin = 24 if with_axes else 0
    legend_h = 36 if with_legend else 0
    w = cols * cell_size + margin
    h = rows * cell_size + margin + legend_h

    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Cells
    for r in range(rows):
        for c in range(cols):
            color = PALETTE[int(arr[r, c])]
            x0 = margin + c * cell_size
            y0 = margin + r * cell_size
            draw.rectangle(
                [x0, y0, x0 + cell_size - 1, y0 + cell_size - 1],
                fill=color,
                outline=(50, 50, 50) if color == (255, 255, 255) else None,
            )

    # Axes
    if with_axes:
        for c in range(0, cols, 5):
            draw.text((margin + c * cell_size + 1, 4), str(c), fill=(0, 0, 0), font=font)
        for r in range(0, rows, 5):
            draw.text((2, margin + r * cell_size + 1), str(r), fill=(0, 0, 0), font=font)

    # Legend strip at bottom
    if with_legend:
        sw = max(8, (w - 16) // 16)
        y0 = margin + rows * cell_size + 4
        for i, color in enumerate(PALETTE):
            x0 = 8 + i * sw
            draw.rectangle([x0, y0, x0 + sw - 2, y0 + 18], fill=color, outline=(0, 0, 0))
            draw.text((x0 + 1, y0 + 20), str(i), fill=(0, 0, 0), font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def grid_to_png_base64(grid: Sequence[Sequence[int]] | np.ndarray, **kw) -> str:
    return base64.b64encode(grid_to_png_bytes(grid, **kw)).decode()


def diff_grids(
    a: Sequence[Sequence[int]] | np.ndarray | None,
    b: Sequence[Sequence[int]] | np.ndarray,
) -> dict:
    """Summarise the change from grid ``a`` to grid ``b``.

    Returns:
        ``{changed_cells, bbox, by_transition}`` where ``bbox`` is
        ``(r0, c0, r1, c1)`` or ``None`` if no change. ``by_transition`` maps
        ``"old->new"`` to count.
    """
    if a is None:
        b_arr = to_numpy(b)
        return {"changed_cells": int(b_arr.size), "bbox": (0, 0, b_arr.shape[0] - 1, b_arr.shape[1] - 1), "by_transition": {}}
    a_arr = to_numpy(a)
    b_arr = to_numpy(b)
    if a_arr.shape != b_arr.shape:
        return {"changed_cells": -1, "bbox": None, "by_transition": {}, "shape_changed": True}
    mask = a_arr != b_arr
    n = int(mask.sum())
    if n == 0:
        return {"changed_cells": 0, "bbox": None, "by_transition": {}}
    rs, cs = np.where(mask)
    bbox = (int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max()))
    transitions: Counter = Counter()
    for r, c in zip(rs.tolist(), cs.tolist()):
        transitions[f"{int(a_arr[r, c])}->{int(b_arr[r, c])}"] += 1
    return {
        "changed_cells": n,
        "bbox": bbox,
        "by_transition": dict(transitions.most_common(10)),
    }


def color_legend_text() -> str:
    """Human-readable color → ASCII char mapping for the prompt."""
    rows = []
    names = [
        "black", "blue", "red", "green", "yellow", "gray", "magenta", "orange",
        "sky", "maroon", "white", "brown", "dgray", "ddgray", "dddgray", "purple",
    ]
    for i, name in enumerate(names):
        rows.append(f"  {i:>2} = {ASCII_CHARS[i]!r}  ({name})")
    return "\n".join(rows)
