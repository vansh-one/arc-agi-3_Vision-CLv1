# vispark-vision-cl-arc-agi-3

> ARC-AGI-3 agent driven by **Vispark/Vision Large** with **Continual
> Learning** weights carried across every game and level. Includes a small
> agentic harness (single action, action sequences, sandboxed python
> analysis, level reset / concede), per-frame multimodal rendering (ASCII +
> labelled PNG), and a fully reproducible two-phase pipeline:
> **Phase 1** explores the public game set offline while CL weights grow,
> **Phase 2** uses those frozen weights to play in
> `OperationMode.COMPETITION` and produce one official scorecard.

[![ARC-AGI-3](https://img.shields.io/badge/benchmark-ARC--AGI--3-blue)](https://arcprize.org/arc-agi/3)
[![Vispark Vision](https://img.shields.io/badge/model-Vispark%20Vision%20Large-orange)](https://api.lab.vispark.in/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Why Continual Learning matters for ARC-AGI-3

ARC-AGI-3 explicitly tests *skill-acquisition efficiency over time*: agents
must build a world model on the fly, transfer reasoning between games and
levels, and avoid re-deriving the same mechanics from scratch each turn.
Vispark Vision's Continual Learning mode is uniquely suited to this — every
call returns a `weights` blob that, when fed into the next call, restores
the latent context the model accumulated in prior turns *across the
1 M-token window*. This is far cheaper than putting the entire transcript
into every prompt, and crucially: the weights carry across game boundaries.

The two-phase setup exploits that property:

```
Phase 1 (offline exploration)
  start with empty weights
  ┌──────────────────────────────────────────┐
  │ for each pass:                           │
  │   for each public game (25 total):       │
  │     play game → weights = new_weights    │
  │     persist weights to disk              │
  │ stop on:                                 │
  │   (a) min per-game score ≥ target, OR    │
  │   (b) weights size has saturated, OR     │
  │   (c) Vision returns "insufficient       │
  │       tokens" (CL window full)           │
  └──────────────────────────────────────────┘
                       │
                       ▼
Phase 2 (online competition)
  load final weights
  open ONE scorecard in COMPETITION mode
  play every game once, only level-resets allowed
  close scorecard → scorecard_url
                       │
                       ▼
  paste scorecard_url into submission.yaml
  open PR against ARC-AGI-Community-Leaderboard
```

## Repository layout

```
final-submission/
├── README.md                       — this file
├── LICENSE                         — MIT
├── pyproject.toml                  — package + CLI entrypoints
├── .env.example                    — secret template
├── docs/
│   └── ARCHITECTURE.md             — deep-dive on the harness
├── submission/
│   ├── submission.yaml             — Community Leaderboard YAML (fill in)
│   └── HOW_TO_SUBMIT.md            — exact submission walk-through
├── src/vision_arc_agi/
│   ├── vision.py                   — Vispark Vision API client (CL + tools + images)
│   ├── arc_runner.py               — wraps the arc-agi toolkit
│   ├── grid.py                     — ASCII + PNG grid rendering
│   ├── state.py                    — per-level history + repetition / stuck warnings
│   ├── memory.py                   — WeightsStore (latest.json + saturation detector)
│   ├── sandbox.py                  — safe python exec for analyze_with_python
│   ├── tools.py                    — tool specs + ToolDecision parser
│   ├── prompts.py                  — system + per-turn prompts
│   ├── agent.py                    — main play_game loop
│   ├── train.py                    — Phase 1 trainer (3-criteria stop)
│   ├── compete.py                  — Phase 2 competition runner
│   └── cli.py                      — argparse entry points
├── scripts/                        — `python scripts/{train,compete,inspect}.py`
├── tests/                          — 33 unit tests + 1 live smoke
├── weights/                        — `latest.json` (gitignored)
├── runs/                           — per-game JSONL transcripts (gitignored)
├── recordings/                     — official toolkit replay files (gitignored)
└── environment_files/              — local game cache (gitignored)
```

## Quick start

### Prerequisites

- Python 3.12+
- `uv` (or `pip`)
- A Vispark Vision API key  →  https://lab.vispark.in/
- An ARC-AGI-3 API key      →  https://arcprize.org/platform

### Install

```bash
cp .env.example .env
# Edit .env, set VISION_API_KEY and ARC_API_KEY
uv sync
```

### Sanity check

```bash
uv run pytest               # 33 unit tests
uv run vision-arc-download  # lists & caches all 25 public games
```

### Phase 1 — train

```bash
uv run vision-arc-train --target 0.80 --max-passes 10
```

Useful flags:

| Flag                          | Default | Meaning                                                        |
|-------------------------------|---------|----------------------------------------------------------------|
| `--games <ids...>`            | all     | filter to specific game ids (prefixes also accepted)           |
| `--target FLOAT`              | 0.80    | stop when every game's est. score reaches this                 |
| `--max-passes INT`            | 10      | safety ceiling on passes                                       |
| `--saturation-lookback INT`   | 4       | size-history window for the saturation detector                |
| `--saturation-rel-delta`      | 0.01    | spread/mean of weight sizes that counts as "saturated"         |
| `--play-size {small|med|large}` | large | Vision size for action decisions                               |
| `--analysis-size`             | medium  | Vision size for `analyze_with_python` sub-calls                |
| `--no-image`                  | (off)   | skip PNG (text-only prompts)                                   |
| `--max-turns-per-game INT`    | none    | hard cap on Vision calls per game (useful for time-boxed runs) |

Output:

- `weights/latest.json` — current CL weights blob + diagnostics
- `weights/session-pass-N.json` — snapshot after each pass
- `runs/<run-id>.jsonl` — per-game transcript (every Vision call + ARC step)
- `runs/training-summary-<ts>.json` — final stop reason + per-pass scores

### Phase 2 — compete

Once you're happy with the training results:

```bash
# Dry run (no publish — scorecard stays open server-side and auto-closes)
uv run vision-arc-compete

# Real publish (requires explicit opt-in)
export ARC_ALLOW_LEADERBOARD_SUBMIT=yes-publish-to-leaderboard
uv run vision-arc-compete --publish \
    --source-url "https://github.com/<you>/vispark-vision-cl-arc-agi-3"
```

## Safety rails

The toolkit refuses to publish a competition scorecard unless
`ARC_ALLOW_LEADERBOARD_SUBMIT=yes-publish-to-leaderboard` is set. This
prevents an accidental `vision-arc-compete --publish` from committing to
the public leaderboard before you've reviewed training.

Training is **OFFLINE** by default (`OperationMode.OFFLINE` after one
NORMAL-mode catalogue fetch) — no training pass ever pings ARC servers
beyond the initial download.

## Scoring intuition

ARC-AGI-3 computes a per-level score of
$(human\_baseline / ai\_actions)^2$, capped at ~1.32, weighted by 1-indexed
level number; the per-game score is the weighted average and the total is
the mean across games. The harness exposes a client-side estimate of this
on every turn so the agent (and you) can see the expected score in real
time. The official figure is whatever `close_scorecard` returns.

## Acknowledgements & links

- ARC-AGI-3 benchmark and toolkit: https://docs.arcprize.org/
- ARC Prize policy: https://arcprize.org/policy
- Community Leaderboard: https://github.com/arcprize/ARC-AGI-Community-Leaderboard
- Vispark Vision API: https://api.lab.vispark.in/

## License

MIT — see [LICENSE](LICENSE).
