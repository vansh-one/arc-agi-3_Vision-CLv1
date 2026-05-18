"""Persist Continual Learning weights + sidecar metadata to disk.

A weights blob is an opaque base64-ish string. We store it together with
diagnostics (cl_usage, length history, last update time, training stats) so
``train.py`` can detect saturation across multiple sessions.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TrainingStats:
    games_played: int = 0
    levels_completed: int = 0
    total_actions: int = 0
    last_session_avg_score: float = 0.0
    last_session_per_game: dict[str, float] = field(default_factory=dict)
    weight_size_history: list[int] = field(default_factory=list)
    cl_usage_history: list[int] = field(default_factory=list)


@dataclass
class WeightsRecord:
    """One persisted weights blob + metadata."""

    blob: str
    cl_usage: int = 0
    updated_at: float = 0.0
    note: str = ""
    stats: TrainingStats = field(default_factory=TrainingStats)

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "WeightsRecord":
        d = json.loads(s)
        d["stats"] = TrainingStats(**(d.get("stats") or {}))
        return cls(**d)


class WeightsStore:
    """A folder of JSON files. ``latest.json`` is the active set."""

    def __init__(self, root: str | os.PathLike = "weights"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # --- paths ----------------------------------------------------- #
    @property
    def latest_path(self) -> Path:
        return self.root / "latest.json"

    def session_path(self, name: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
        return self.root / f"session-{safe}.json"

    # --- read/write ----------------------------------------------- #
    def load_latest(self) -> WeightsRecord | None:
        if not self.latest_path.exists():
            return None
        return WeightsRecord.from_json(self.latest_path.read_text())

    def save(self, record: WeightsRecord, *, also_session: str | None = None):
        record.updated_at = time.time()
        self.latest_path.write_text(record.to_json())
        if also_session:
            self.session_path(also_session).write_text(record.to_json())

    # --- saturation detector ------------------------------------- #
    @staticmethod
    def is_saturated(history: list[int], *, lookback: int = 4, rel_delta: float = 0.01) -> bool:
        """True if the last ``lookback`` weights sizes vary by less than ``rel_delta``
        relative to their mean.

        Used as one of the three training stop criteria.
        """
        if len(history) < lookback:
            return False
        recent = history[-lookback:]
        mean = sum(recent) / len(recent)
        if mean == 0:
            return False
        spread = (max(recent) - min(recent)) / mean
        return spread < rel_delta
