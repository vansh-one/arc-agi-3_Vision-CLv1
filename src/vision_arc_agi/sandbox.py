"""Tiny Python sandbox for the ``analyze_with_python`` tool.

The agent occasionally needs to compute things on the grid (find clusters,
distance to a target, symmetry checks). Letting Vision write a python snippet
is far more flexible than pre-shipping a hundred analysis tools. We constrain
the namespace and capture stdout with a hard cap.

Hard constraints:

- no imports beyond a whitelist
- no ``exec`` / ``compile`` / ``__import__`` / file I/O
- 4 KB stdout cap, 2-second CPU cap
- numpy + collections + math + itertools available

The sandbox is *not* a hostile-code firewall — it's a footgun reducer for our
own LLM. Running this inside competition mode against ARC servers is fine
because the snippet never reaches ARC; only the analysis output influences the
agent's next decision.
"""

from __future__ import annotations

import io
import math
import signal
import sys
import textwrap
from collections import Counter, defaultdict, deque
from contextlib import redirect_stderr, redirect_stdout
from itertools import chain, combinations, groupby, permutations, product
from typing import Any

import numpy as np

OUTPUT_LIMIT = 4000  # chars

SAFE_GLOBALS: dict[str, Any] = {
    "__builtins__": {
        "abs": abs, "all": all, "any": any, "bool": bool, "bytes": bytes,
        "chr": chr, "dict": dict, "divmod": divmod, "enumerate": enumerate,
        "filter": filter, "float": float, "frozenset": frozenset, "hex": hex,
        "int": int, "isinstance": isinstance, "iter": iter, "len": len,
        "list": list, "map": map, "max": max, "min": min, "next": next,
        "oct": oct, "ord": ord, "pow": pow, "print": print, "range": range,
        "repr": repr, "reversed": reversed, "round": round, "set": set,
        "slice": slice, "sorted": sorted, "str": str, "sum": sum,
        "tuple": tuple, "type": type, "zip": zip, "True": True, "False": False,
        "None": None,
    },
    "np": np,
    "numpy": np,
    "math": math,
    "Counter": Counter,
    "defaultdict": defaultdict,
    "deque": deque,
    "chain": chain,
    "combinations": combinations,
    "groupby": groupby,
    "permutations": permutations,
    "product": product,
}


class SandboxTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise SandboxTimeout("snippet exceeded 2s")


def run_python_snippet(
    code: str,
    *,
    grid: np.ndarray | None = None,
    timeout_seconds: int = 2,
    extra_locals: dict[str, Any] | None = None,
) -> dict[str, str | bool]:
    """Execute ``code`` in a sandboxed namespace with ``grid`` available.

    Returns ``{"ok": bool, "stdout": str, "error": str}``.
    """
    locals_dict: dict[str, Any] = {}
    if grid is not None:
        locals_dict["grid"] = np.asarray(grid).copy()
        locals_dict["G"] = locals_dict["grid"]
    if extra_locals:
        locals_dict.update(extra_locals)

    stdout = io.StringIO()
    stderr = io.StringIO()
    error = ""

    prev_handler = None
    try:
        if hasattr(signal, "SIGALRM"):  # POSIX
            prev_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout_seconds)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(textwrap.dedent(code), SAFE_GLOBALS, locals_dict)  # noqa: S102
    except SandboxTimeout as e:
        error = f"TIMEOUT: {e}"
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            if prev_handler is not None:
                signal.signal(signal.SIGALRM, prev_handler)

    out = stdout.getvalue() or stderr.getvalue()
    truncated = False
    if len(out) > OUTPUT_LIMIT:
        out = out[:OUTPUT_LIMIT] + f"\n[...truncated, {len(out) - OUTPUT_LIMIT} more chars]"
        truncated = True

    return {
        "ok": not bool(error),
        "stdout": out,
        "error": error,
        "truncated": truncated,
    }


# Pre-built helpers the model can call inside the snippet without re-implementing.
# Made available by injecting into SAFE_GLOBALS at module import.

def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    if not mask.any():
        return None
    rs, cs = np.where(mask)
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def find_color(g: np.ndarray, color: int) -> list[tuple[int, int]]:
    rs, cs = np.where(g == color)
    return list(zip(rs.tolist(), cs.tolist()))


def flood_clusters(g: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    """4-connected clusters of cells with given color."""
    h, w = g.shape
    seen = np.zeros_like(g, dtype=bool)
    clusters: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or g[r, c] != color:
                continue
            stack = [(r, c)]
            cluster: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                if 0 <= y < h and 0 <= x < w and not seen[y, x] and g[y, x] == color:
                    seen[y, x] = True
                    cluster.append((y, x))
                    stack.extend([(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)])
            clusters.append(cluster)
    return clusters


SAFE_GLOBALS["find_color"] = find_color
SAFE_GLOBALS["flood_clusters"] = flood_clusters
SAFE_GLOBALS["bbox"] = _bbox
