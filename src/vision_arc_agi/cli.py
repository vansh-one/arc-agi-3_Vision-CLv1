"""Command-line entrypoints registered in pyproject.toml.

Each function expects the user's ``.env`` to have populated ``VISION_API_KEY``
and ``ARC_API_KEY``. We use ``python-dotenv`` if it's importable so people
running ``uv run`` from the project root pick up the local ``.env``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .arc_runner import ArcRunner
from .compete import CompetitionConfig, run_competition
from .memory import WeightsStore
from .train import TrainConfig, run_training


def _load_dotenv():
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass


def _setup_logging(verbose: bool):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# ------------------------------------------------------------------ #
def train_main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    p = argparse.ArgumentParser(
        prog="vision-arc-train",
        description="Phase 1: explore + learn ARC-AGI-3 with Vision (Large) + Continual Learning.",
    )
    p.add_argument("--games", nargs="*", default=[], help="game id prefixes/titles to include (default: all)")
    p.add_argument("--max-passes", type=int, default=10)
    p.add_argument("--target", type=float, default=0.80, help="per-game stop threshold")
    p.add_argument("--saturation-lookback", type=int, default=4)
    p.add_argument("--saturation-rel-delta", type=float, default=0.01)
    p.add_argument("--play-size", default="large", choices=["small", "medium", "large"])
    p.add_argument("--analysis-size", default="medium", choices=["small", "medium", "large"])
    p.add_argument("--no-image", action="store_true", help="omit PNG frame from prompt")
    p.add_argument("--png-cell", type=int, default=8)
    p.add_argument("--max-turns-per-game", type=int, default=None)
    p.add_argument("--weights-dir", default="weights")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--environments-dir", default="environment_files")
    p.add_argument("--recordings-dir", default="recordings")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    _setup_logging(args.verbose)

    cfg = TrainConfig(
        weights_dir=args.weights_dir,
        runs_dir=args.runs_dir,
        environments_dir=args.environments_dir,
        recordings_dir=args.recordings_dir,
        play_size=args.play_size,
        analysis_size=args.analysis_size,
        games=args.games,
        max_passes=args.max_passes,
        per_game_target=args.target,
        saturation_lookback=args.saturation_lookback,
        saturation_rel_delta=args.saturation_rel_delta,
        send_image=not args.no_image,
        png_cell=args.png_cell,
        max_turns_per_game=args.max_turns_per_game,
    )
    summary = run_training(cfg)
    print(json.dumps(summary, indent=2))
    return 0


def compete_main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    p = argparse.ArgumentParser(
        prog="vision-arc-compete",
        description="Phase 2: run the trained agent in COMPETITION mode on ARC-AGI-3.",
    )
    p.add_argument("--games", nargs="*", default=[])
    p.add_argument("--tags", nargs="*", default=["vispark-vision-cl-large", "arc-agi-3"])
    p.add_argument("--source-url", default="", help="public link to this repo / scorecard description")
    p.add_argument("--publish", action="store_true",
                   help="close scorecard at end (requires ARC_ALLOW_LEADERBOARD_SUBMIT)")
    p.add_argument("--play-size", default="large", choices=["small", "medium", "large"])
    p.add_argument("--analysis-size", default="medium", choices=["small", "medium", "large"])
    p.add_argument("--no-image", action="store_true")
    p.add_argument("--png-cell", type=int, default=8)
    p.add_argument("--max-turns-per-game", type=int, default=None)
    p.add_argument("--weights-dir", default="weights")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--environments-dir", default="environment_files")
    p.add_argument("--recordings-dir", default="recordings")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    _setup_logging(args.verbose)

    cfg = CompetitionConfig(
        weights_dir=args.weights_dir,
        runs_dir=args.runs_dir,
        environments_dir=args.environments_dir,
        recordings_dir=args.recordings_dir,
        play_size=args.play_size,
        analysis_size=args.analysis_size,
        games=args.games,
        tags=args.tags,
        source_url=args.source_url,
        publish=args.publish,
        send_image=not args.no_image,
        png_cell=args.png_cell,
        max_turns_per_game=args.max_turns_per_game,
    )
    summary = run_competition(cfg)
    print(json.dumps(summary, indent=2, default=str))
    return 0


def inspect_main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    p = argparse.ArgumentParser(
        prog="vision-arc-inspect",
        description="Inspect saved Continual Learning weights.",
    )
    p.add_argument("--weights-dir", default="weights")
    p.add_argument("--session", default=None, help="session name (eg. pass-3)")
    args = p.parse_args(argv)

    store = WeightsStore(args.weights_dir)
    rec = (
        __import__("json").loads(store.session_path(args.session).read_text())
        if args.session
        else (store.load_latest().to_json() if store.load_latest() else None)
    )
    if not rec:
        print("No weights record found.")
        return 1
    if isinstance(rec, dict):
        rec_d = rec
    else:
        rec_d = __import__("json").loads(rec)
    # Don't dump the giant blob to stdout — show metadata only.
    rec_d_meta = {**rec_d, "blob": f"<{len(rec_d.get('blob') or '')} chars>"}
    print(json.dumps(rec_d_meta, indent=2, default=str))
    return 0


def download_main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    p = argparse.ArgumentParser(
        prog="vision-arc-download",
        description="Fetch the public game catalogue + cache local env files.",
    )
    p.add_argument("--environments-dir", default="environment_files")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    _setup_logging(args.verbose)

    api_key = os.environ.get("ARC_API_KEY")
    if not api_key:
        print("ARC_API_KEY not set", file=sys.stderr)
        return 1
    runner = ArcRunner(
        mode="NORMAL",
        api_key=api_key,
        base_url=os.environ.get("ARC_BASE_URL"),
        environments_dir=args.environments_dir,
        save_recording=False,
    )
    games = runner.list_games()
    print(f"{len(games)} games:")
    for g in games:
        print(f"  {g.game_id}  {g.title:8s}  tags={','.join(g.tags) or '-':<16}"
              f"  levels={len(g.baseline_actions)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    rest = sys.argv[2:]
    if cmd == "train":
        sys.exit(train_main(rest))
    if cmd == "compete":
        sys.exit(compete_main(rest))
    if cmd == "inspect":
        sys.exit(inspect_main(rest))
    if cmd == "download":
        sys.exit(download_main(rest))
    print(f"Unknown command: {cmd}. Try train|compete|inspect|download")
    sys.exit(2)
