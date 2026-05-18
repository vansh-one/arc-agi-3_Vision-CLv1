"""Phase 2: Competition-mode run using trained weights.

Loads the latest weights from disk and plays each public game once in
``OperationMode.COMPETITION``. Per the ARC docs:

- Only ONE scorecard is opened
- ``make()`` is called once per environment
- Only LEVEL resets are permitted (no game resets)
- ``get_scorecard`` is disabled mid-run

After all games complete the scorecard is closed, which is the act that
publishes results to the ARC scorecard URL (``arcprize.org/scorecards/<id>``).
We refuse to close unless ``ARC_ALLOW_LEADERBOARD_SUBMIT=yes-publish-to-leaderboard``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .agent import GameResult, VisionAgent
from .arc_runner import ArcRunner
from .memory import WeightsStore
from .vision import VisionClient, VisionSize

log = logging.getLogger(__name__)


@dataclass
class CompetitionConfig:
    weights_dir: str = "weights"
    runs_dir: str = "runs"
    environments_dir: str = "environment_files"
    recordings_dir: str = "recordings"
    play_size: VisionSize = "large"
    analysis_size: VisionSize = "medium"
    games: list[str] = field(default_factory=list)  # empty → ALL
    tags: list[str] = field(default_factory=lambda: ["vispark-vision-cl-large", "arc-agi-3"])
    source_url: str = ""
    publish: bool = False
    send_image: bool = True
    png_cell: int = 8
    max_turns_per_game: int | None = None


# --------------------------------------------------------------------- #
def run_competition(cfg: CompetitionConfig) -> dict[str, Any]:
    console = Console()
    store = WeightsStore(cfg.weights_dir)
    record = store.load_latest()
    if record is None or not record.blob:
        raise RuntimeError(
            "No trained weights at weights/latest.json. Run training first: "
            "`vision-arc-train`."
        )

    console.print(
        f"[green]Loaded weights[/green] len={len(record.blob)} "
        f"cl_usage={record.cl_usage} stats.games_played={record.stats.games_played}"
    )

    arc_key = os.environ.get("ARC_API_KEY")
    arc_url = os.environ.get("ARC_BASE_URL")
    if not arc_key:
        raise RuntimeError("ARC_API_KEY required for competition mode")

    arc = ArcRunner(
        mode="COMPETITION",
        api_key=arc_key, base_url=arc_url,
        environments_dir=cfg.environments_dir,
        recordings_dir=cfg.recordings_dir,
        save_recording=True,
    )

    games_all = arc.list_games()
    games = [g for g in games_all if not cfg.games or any(
        p in g.game_id or p.lower() == g.title.lower() for p in cfg.games
    )]
    if not games:
        raise RuntimeError(f"No games matched filter {cfg.games}")
    console.print(f"[bold]Competition: {len(games)} game(s)[/bold]")

    scorecard_id = arc.open_scorecard(tags=cfg.tags, source_url=cfg.source_url)
    console.print(f"[bold]Opened scorecard[/bold] {scorecard_id}")

    results: dict[str, GameResult] = {}
    final_weights = record.blob

    try:
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
            for i, gi in enumerate(games, 1):
                console.print(
                    f"[cyan]Competition {i}/{len(games)}:"
                    f" {gi.game_id} ({gi.title})[/cyan]"
                )
                try:
                    run = agent.play_game(
                        gi,
                        weights=final_weights,
                        phase="compete",
                        attempt=1,
                    )
                except Exception as e:  # noqa: BLE001
                    log.exception("Game crashed in competition")
                    console.print(f"[red]Game crashed:[/red] {e}")
                    continue
                if run.final_weights:
                    # Keep updated weights forward across games (helps later games)
                    final_weights = run.final_weights
                results[gi.game_id] = run.result
                console.print(
                    f"  → state={run.result.state} "
                    f"completed={run.result.levels_completed}/{run.result.total_levels} "
                    f"actions={run.result.actions_used} "
                    f"est_score={run.result.estimated_game_score:.3f}"
                )
    finally:
        if cfg.publish:
            console.print("[bold yellow]Closing scorecard (publishing)…[/bold yellow]")
            try:
                summary = arc.close_scorecard(force_publish=False)
            except RuntimeError as e:
                console.print(f"[red]Refused to publish:[/red] {e}")
                summary = {"refused": str(e)}
        else:
            console.print(
                "[yellow]Skipping scorecard close (--no-publish). "
                "The scorecard remains open and will auto-close on the ARC server"
                " after 15 minutes of inactivity.[/yellow]"
            )
            summary = {"scorecard_id": scorecard_id, "left_open": True}

    final = {
        "scorecard_id": scorecard_id,
        "scorecard_url": f"https://arcprize.org/scorecards/{scorecard_id}",
        "summary": summary,
        "per_game": {gid: r.to_dict() for gid, r in results.items()},
    }
    Path(cfg.runs_dir, f"competition-{int(time.time())}.json").write_text(
        json.dumps(final, indent=2)
    )
    _render_competition_table(console, results, final["scorecard_url"])
    return final


def _render_competition_table(console: Console, results: dict[str, GameResult], url: str):
    t = Table(title="Competition results (estimated client-side scores)")
    t.add_column("game_id")
    t.add_column("title")
    t.add_column("state")
    t.add_column("completed", justify="right")
    t.add_column("actions", justify="right")
    t.add_column("est_score", justify="right")
    for gid, r in results.items():
        t.add_row(
            gid, r.title, r.state,
            f"{r.levels_completed}/{r.total_levels}",
            str(r.actions_used),
            f"{r.estimated_game_score:.3f}",
        )
    console.print(t)
    if results:
        avg = sum(r.estimated_game_score for r in results.values()) / len(results)
        console.print(f"[bold]Mean estimated score: {avg:.3f}[/bold]")
    console.print(f"[bold green]Scorecard URL:[/bold green] {url}")
