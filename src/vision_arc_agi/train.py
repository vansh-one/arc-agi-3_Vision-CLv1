"""Phase 1: Exploration / Training with Continual Learning.

Runs Vision (Large for play, Medium for analysis) over the local games. The
Continual Learning weights blob is carried forward across games — every game
sees a (slightly) more capable agent.

Stops on ANY of the three user-specified criteria:

1. ``per_game_target`` — every game in the last pass got ≥ ``target`` estimated
   score (default 0.80) → we believe Vision has learned the public games.
2. ``saturation`` — weights size has stayed within ``rel_delta`` for
   ``lookback`` consecutive sessions → CL is no longer absorbing new info.
3. ``insufficient_tokens`` — Vision API raised ``VisionInsufficientTokensError``
   at any point → the CL blob has saturated the model's 1 M-token window.

A hard pass ceiling of ``max_passes`` is also enforced as a safety net.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .agent import GameResult, VisionAgent
from .arc_runner import ArcRunner, GameInfo
from .memory import TrainingStats, WeightsRecord, WeightsStore
from .vision import VisionClient, VisionInsufficientTokensError, VisionSize

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    weights_dir: str = "weights"
    runs_dir: str = "runs"
    environments_dir: str = "environment_files"
    recordings_dir: str = "recordings"
    play_size: VisionSize = "large"
    analysis_size: VisionSize = "medium"
    games: list[str] = field(default_factory=list)        # empty → all
    max_passes: int = 10
    per_game_target: float = 0.80
    saturation_lookback: int = 4
    saturation_rel_delta: float = 0.01
    send_image: bool = True
    png_cell: int = 8
    download_only_if_missing: bool = True
    max_turns_per_game: int | None = None


@dataclass
class PassResult:
    pass_index: int
    started_at: float
    per_game: dict[str, GameResult] = field(default_factory=dict)
    weights_len: int = 0
    cl_usage: int = 0

    @property
    def avg_score(self) -> float:
        if not self.per_game:
            return 0.0
        return sum(r.estimated_game_score for r in self.per_game.values()) / len(self.per_game)

    def min_score(self) -> float:
        if not self.per_game:
            return 0.0
        return min(r.estimated_game_score for r in self.per_game.values())


# --------------------------------------------------------------------- #
def _ensure_games_local(env_dir: str, api_key: str | None, base_url: str | None) -> list[GameInfo]:
    """Use ONLINE mode briefly to fetch the game catalogue + download any
    missing local environment files, then return the catalogue. Subsequent
    passes can then run fully OFFLINE.
    """
    log.info("Downloading game catalogue + missing local environments")
    online = ArcRunner(
        mode="NORMAL",
        api_key=api_key, base_url=base_url,
        environments_dir=env_dir, save_recording=False,
    )
    games = online.list_games()
    log.info("Catalogue: %d games", len(games))
    # Touch each env so the toolkit downloads the assets locally.
    for g in games:
        try:
            env = online.arc.make(game_id=g.game_id, save_recording=False)
            if env is not None:
                # Don't actually play — just trigger the download / cache.
                pass
        except Exception as e:  # noqa: BLE001
            log.warning("Failed to pre-fetch %s: %s", g.game_id, e)
    return games


def _filter(games: list[GameInfo], pattern: list[str]) -> list[GameInfo]:
    if not pattern:
        return games
    keep: list[GameInfo] = []
    for g in games:
        for p in pattern:
            if p in g.game_id or p.lower() == g.title.lower():
                keep.append(g)
                break
    return keep


# --------------------------------------------------------------------- #
def run_training(cfg: TrainConfig) -> dict[str, Any]:
    console = Console()
    store = WeightsStore(cfg.weights_dir)

    # Load prior weights if any
    record = store.load_latest()
    if record is None:
        record = WeightsRecord(blob="", cl_usage=0, note="initial empty weights")
        console.print("[yellow]Starting from EMPTY weights.[/yellow]")
    else:
        console.print(
            f"[green]Loaded prior weights[/green] "
            f"len={len(record.blob)} cl_usage={record.cl_usage} "
            f"games_played={record.stats.games_played}"
        )

    arc_key = os.environ.get("ARC_API_KEY")
    arc_url = os.environ.get("ARC_BASE_URL")

    # Pre-fetch games (NORMAL once, then OFFLINE).
    games_all = _ensure_games_local(cfg.environments_dir, arc_key, arc_url)
    games = _filter(games_all, cfg.games)
    if not games:
        raise RuntimeError(f"No games matched filter {cfg.games}")
    console.print(f"[bold]Training over {len(games)} game(s):[/bold] "
                  f"{', '.join(g.game_id for g in games)}")

    arc = ArcRunner(
        mode="OFFLINE",
        api_key=arc_key, base_url=arc_url,
        environments_dir=cfg.environments_dir,
        recordings_dir=cfg.recordings_dir,
        save_recording=True,
    )

    pass_results: list[PassResult] = []
    stop_reason: str | None = None

    with VisionClient() as vision:
        agent = VisionAgent(
            vision=vision, arc=arc,
            play_size=cfg.play_size,
            analysis_size=cfg.analysis_size,
            runs_dir=cfg.runs_dir,
            send_image=cfg.send_image,
            png_cell=cfg.png_cell,
            max_turns=cfg.max_turns_per_game,
        )
        for pass_idx in range(1, cfg.max_passes + 1):
            console.rule(f"[bold]Pass {pass_idx} / {cfg.max_passes}")
            pr = PassResult(pass_index=pass_idx, started_at=time.time())
            for gi, ginfo in enumerate(games, 1):
                console.print(
                    f"[cyan]Pass {pass_idx} – game {gi}/{len(games)}:"
                    f" {ginfo.game_id} ({ginfo.title})[/cyan]"
                )
                try:
                    run = agent.play_game(
                        ginfo,
                        weights=record.blob or None,
                        phase=f"train-p{pass_idx}",
                        attempt=pass_idx,
                    )
                except VisionInsufficientTokensError as e:
                    stop_reason = f"insufficient_tokens: {e}"
                    console.print(f"[red]STOP[/red] insufficient_tokens — {e}")
                    break
                if run.final_weights:
                    record = WeightsRecord(
                        blob=run.final_weights,
                        cl_usage=run.final_cl_usage,
                        note=f"after game {ginfo.game_id} pass {pass_idx}",
                        stats=record.stats,
                    )
                    record.stats.games_played += 1
                    record.stats.weight_size_history.append(len(record.blob))
                    record.stats.cl_usage_history.append(record.cl_usage)
                    store.save(record)
                pr.per_game[ginfo.game_id] = run.result
                console.print(
                    f"  → state={run.result.state} "
                    f"completed={run.result.levels_completed}/{run.result.total_levels} "
                    f"actions={run.result.actions_used} "
                    f"est_score={run.result.estimated_game_score:.3f} "
                    f"weights_len={len(record.blob)} cl={record.cl_usage}"
                )

            if stop_reason:
                break

            pr.weights_len = len(record.blob)
            pr.cl_usage = record.cl_usage
            pass_results.append(pr)
            record.stats.last_session_avg_score = pr.avg_score
            record.stats.last_session_per_game = {
                gid: r.estimated_game_score for gid, r in pr.per_game.items()
            }
            store.save(record, also_session=f"pass-{pass_idx}")

            _render_pass_table(console, pass_idx, pr)

            # Stop-criteria checks
            min_score = pr.min_score()
            if min_score >= cfg.per_game_target:
                stop_reason = (
                    f"per_game_target reached (min_score={min_score:.3f} ≥ "
                    f"{cfg.per_game_target})"
                )
                break
            if WeightsStore.is_saturated(
                record.stats.weight_size_history,
                lookback=cfg.saturation_lookback,
                rel_delta=cfg.saturation_rel_delta,
            ):
                stop_reason = (
                    f"weights saturated over last "
                    f"{cfg.saturation_lookback} updates "
                    f"(rel_delta < {cfg.saturation_rel_delta})"
                )
                break

    if stop_reason is None:
        stop_reason = f"max_passes={cfg.max_passes} reached"

    console.print(f"[bold green]TRAINING COMPLETE.[/bold green] stop_reason: {stop_reason}")
    summary = {
        "stop_reason": stop_reason,
        "passes": [
            {
                "pass": p.pass_index,
                "avg_score": p.avg_score,
                "min_score": p.min_score(),
                "weights_len": p.weights_len,
                "cl_usage": p.cl_usage,
                "per_game": {gid: r.estimated_game_score for gid, r in p.per_game.items()},
            }
            for p in pass_results
        ],
        "final_weights_len": len(record.blob),
        "final_cl_usage": record.cl_usage,
        "final_weights_path": str(Path(cfg.weights_dir) / "latest.json"),
    }
    Path(cfg.runs_dir, f"training-summary-{int(time.time())}.json").write_text(
        __import__("json").dumps(summary, indent=2)
    )
    return summary


def _render_pass_table(console: Console, pass_idx: int, pr: PassResult):
    t = Table(title=f"Pass {pass_idx} — avg {pr.avg_score:.3f} min {pr.min_score():.3f}")
    t.add_column("game_id")
    t.add_column("title")
    t.add_column("state")
    t.add_column("completed", justify="right")
    t.add_column("actions", justify="right")
    t.add_column("est_score", justify="right")
    for gid, r in pr.per_game.items():
        t.add_row(
            gid, r.title, r.state,
            f"{r.levels_completed}/{r.total_levels}",
            str(r.actions_used),
            f"{r.estimated_game_score:.3f}",
        )
    console.print(t)
