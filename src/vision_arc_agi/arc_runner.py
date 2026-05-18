"""Thin wrapper around the ``arc-agi`` toolkit.

Handles:
- Operation-mode selection (OFFLINE / NORMAL / ONLINE / COMPETITION)
- Downloading environments on first run (so OFFLINE can take over)
- Per-game session lifecycle (open → many step()s → done)
- Translating our internal action codes into ``GameAction`` enum + data dict
- Safe scorecard close that requires an explicit env flag in competition mode
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

# arc_agi imports lazy / guarded so import-time isn't a hard requirement.
log = logging.getLogger(__name__)


_PUBLISH_GUARD = "ARC_ALLOW_LEADERBOARD_SUBMIT"
_PUBLISH_OK_VALUE = "yes-publish-to-leaderboard"


@dataclass
class GameInfo:
    game_id: str
    title: str
    tags: list[str]
    baseline_actions: list[int]


def _import_arc():
    try:
        from arc_agi import Arcade, OperationMode  # type: ignore
        from arcengine import GameAction, GameState as _GS  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "arc-agi toolkit is not installed. Run `uv sync` or "
            "`pip install arc-agi>=0.9.8` first."
        ) from e
    return Arcade, GameAction, _GS, OperationMode


class ArcRunner:
    """Single-mode driver around the arc-agi toolkit."""

    OFFLINE = "OFFLINE"
    NORMAL = "NORMAL"
    ONLINE = "ONLINE"
    COMPETITION = "COMPETITION"

    def __init__(
        self,
        mode: str = "OFFLINE",
        api_key: str | None = None,
        base_url: str | None = None,
        environments_dir: str = "environment_files",
        recordings_dir: str = "recordings",
        save_recording: bool = True,
        logger: logging.Logger | None = None,
    ):
        Arcade, GameAction, _GS, OperationMode = _import_arc()
        self._GameAction = GameAction
        self._GameState = _GS

        mode_map = {
            "OFFLINE": OperationMode.OFFLINE,
            "NORMAL": OperationMode.NORMAL,
            "ONLINE": OperationMode.ONLINE,
            "COMPETITION": OperationMode.COMPETITION,
        }
        if mode not in mode_map:
            raise ValueError(f"unknown mode {mode}")
        self.mode = mode
        self._is_competition = mode == "COMPETITION"

        kwargs: dict[str, Any] = {
            "operation_mode": mode_map[mode],
            "environments_dir": environments_dir,
            "recordings_dir": recordings_dir,
        }
        if api_key:
            kwargs["arc_api_key"] = api_key
        if base_url:
            kwargs["arc_base_url"] = base_url
        if logger:
            kwargs["logger"] = logger

        self.arc = Arcade(**kwargs)
        self.save_recording = save_recording
        self._scorecard_id: str | None = None

    # ----- discovery ------------------------------------------------ #
    def list_games(self) -> list[GameInfo]:
        envs = self.arc.get_environments() or []
        out: list[GameInfo] = []
        for e in envs:
            out.append(
                GameInfo(
                    game_id=getattr(e, "game_id", ""),
                    title=getattr(e, "title", "") or "",
                    tags=list(getattr(e, "tags", None) or []),
                    baseline_actions=list(getattr(e, "baseline_actions", None) or []),
                )
            )
        return out

    # ----- scorecard ------------------------------------------------ #
    def open_scorecard(self, tags: list[str] | None = None, source_url: str = "") -> str:
        sid = self.arc.create_scorecard(source_url=source_url or None, tags=tags or [])
        self._scorecard_id = sid
        return sid

    def close_scorecard(self, force_publish: bool = False) -> dict[str, Any]:
        """Close the current scorecard.

        In competition mode this is the act that pushes results to the ARC
        leaderboard ingest pipeline. We refuse to do this unless the user has
        set ``ARC_ALLOW_LEADERBOARD_SUBMIT=yes-publish-to-leaderboard`` (or
        ``force_publish=True``).
        """
        if self._is_competition and not force_publish:
            allow = os.environ.get(_PUBLISH_GUARD)
            if allow != _PUBLISH_OK_VALUE:
                raise RuntimeError(
                    f"Refusing to close competition scorecard: "
                    f"set {_PUBLISH_GUARD}={_PUBLISH_OK_VALUE} or pass "
                    f"force_publish=True. Current value: {allow!r}"
                )
        try:
            res = self.arc.close_scorecard(scorecard_id=self._scorecard_id)
        except TypeError:
            # Older toolkit signature
            res = self.arc.close_scorecard(self._scorecard_id)
        if hasattr(res, "model_dump"):
            return res.model_dump()
        if isinstance(res, dict):
            return res
        return {"raw": str(res)}

    def get_scorecard(self) -> dict[str, Any]:
        if self._is_competition:
            return {"unavailable": "competition mode disables get_scorecard"}
        try:
            res = self.arc.get_scorecard(scorecard_id=self._scorecard_id)
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}
        if hasattr(res, "model_dump"):
            return res.model_dump()
        return res if isinstance(res, dict) else {"raw": str(res)}

    # ----- per-game session ---------------------------------------- #
    def make(self, game_id: str):
        return self.arc.make(
            game_id=game_id,
            scorecard_id=self._scorecard_id,
            save_recording=self.save_recording,
        )

    # ----- action translation -------------------------------------- #
    def action_for(self, code: int):
        """Map our 1-7 integer to the toolkit GameAction enum."""
        ga = self._GameAction
        return {
            0: ga.RESET,
            1: ga.ACTION1,
            2: ga.ACTION2,
            3: ga.ACTION3,
            4: ga.ACTION4,
            5: ga.ACTION5,
            6: ga.ACTION6,
            7: ga.ACTION7,
        }[int(code)]

    def state_is_terminal(self, state_value: Any) -> bool:
        return _state_name(state_value) in ("WIN", "GAME_OVER")

    def state_is_win(self, state_value: Any) -> bool:
        return _state_name(state_value) == "WIN"


def _state_name(state_value: Any) -> str:
    """Robustly extract the bare state name regardless of whether the toolkit
    hands us the str-enum (``GameState.WIN``), its ``.value`` (``'WIN'``), or
    ``str(...)`` of either form (``'GameState.WIN'``)."""
    if hasattr(state_value, "name"):
        return str(state_value.name).upper()
    s = str(state_value).upper()
    if "." in s:
        s = s.rsplit(".", 1)[-1]
    return s
