# Architecture deep-dive

This document explains *how* the agent works, in enough detail to debug or
extend it. For "what" and "why", see the top-level `README.md`.

## 1. The control loop in one picture

```
┌──────────────────────────────────────────────────────────────────────┐
│  VisionAgent.play_game(game_info, weights)                          │
│                                                                      │
│   env = ArcRunner.make(game_id)                                      │
│   obs = env.reset()                                                  │
│   state.update_frame(obs)                                            │
│                                                                      │
│   while not terminal:                                                │
│     if no_change_streak ≥ NO_CHANGE_AUTO_RESET → env.reset()         │
│     if level_actions ≥ ceiling                 → env.reset()         │
│                                                                      │
│     content = [ text(initial?), text(turn), image(PNG) ]             │
│     resp    = Vision.call(content, CL=true, weights=W, tools=TOOL_DEFS)
│     W       = resp.weights                                           │
│     d       = parse_tool_call(resp.tool_calls[0])                    │
│                                                                      │
│     if d.kind == 'analysis':                                         │
│         out  = sandbox.run_python(d.code, grid)                      │
│         content = [ text(analysis result) ]    # loop, no action yet │
│         continue                                                     │
│                                                                      │
│     if d.kind == 'action':       env.step(d.action, x, y, reasoning) │
│     if d.kind == 'sequence':     for a in d.sequence: env.step(a, …) │
│     if d.kind == 'reset':        env.reset()                         │
│     if d.kind == 'concede':      env.reset()  (and bail this level)  │
│                                                                      │
│     state.record_step(...)                                           │
│     state.update_frame(obs')                                         │
│                                                                      │
│   summarise → GameResult                                             │
│   return AgentRun(final_weights=W, result=...)                       │
└──────────────────────────────────────────────────────────────────────┘
```

## 2. Modules

| Module           | Concrete responsibilities                                                 |
|------------------|---------------------------------------------------------------------------|
| `vision.py`      | Single Vispark `POST` call. Handles 429/5xx backoff, parses tool_calls vs text, recognises `insufficient tokens`, returns a `VisionResponse` with weights, cl_usage, units. |
| `arc_runner.py`  | Tiny wrapper around the `arc_agi.Arcade` toolkit. Owns OperationMode selection, scorecard open/close (refuses to publish without env opt-in), maps 0..7 → `GameAction`, normalises state names. |
| `grid.py`        | 64x64 grid renderers: ASCII (single char per cell, axes optional) and PNG (cell-size configurable, axes + 16-color legend). Plus diffing, hashing, histograms. |
| `state.py`       | Per-level history. Maintains a state→action memo so the agent can avoid trying the same `(action, x, y)` from the same grid hash twice. Surfaces repetition / stuck / counter warnings. Computes a client-side per-level score estimate. |
| `memory.py`      | `WeightsStore` saves/loads `latest.json` and `session-*.json`. `is_saturated()` is the size-history-based stop criterion. |
| `sandbox.py`     | Safe `exec` for `analyze_with_python`. Whitelisted builtins only, numpy + Counter + helpers, 2 s CPU cap, 4 KB stdout cap. |
| `tools.py`       | The five tool specs Vision sees, plus `parse_tool_call(name, args)` → `ToolDecision`. |
| `prompts.py`     | One static system prompt + two per-turn message builders (initial vs subsequent). |
| `agent.py`       | The play loop above; per-level auto-reset and reset-budget tracking; client-side score estimation. |
| `train.py`       | Pass-loop wrapping `agent.play_game()`; carries weights forward across games; implements the **three stop criteria**: per-game target / size saturation / insufficient-tokens. |
| `compete.py`     | Same loop with `OperationMode.COMPETITION` and `--publish` gate. |
| `cli.py`         | `vision-arc-{train,compete,inspect,download}` entry points. |

## 3. Continual Learning lifecycle

A single CL weights blob lives at `weights/latest.json`. After each game in
Phase 1:

1. The Vision call's response `weights` field is unconditionally captured.
2. A `WeightsRecord` is written atomically to `weights/latest.json`.
3. Every pass also snapshots a copy at `weights/session-pass-N.json`.

Carryforward is intentional: weights flow *into* the next call and *out of*
each call, so by the end of Phase 1 they encode latent skill across all
public games. In Phase 2 the same blob seeds every game so Vision starts
each environment already "warm".

## 4. Stop criteria detail (Phase 1)

`train.run_training` checks after every pass:

```python
min_score = pass_result.min_score()
if min_score >= cfg.per_game_target:
    stop_reason = "per_game_target reached"

elif WeightsStore.is_saturated(
        record.stats.weight_size_history,
        lookback=cfg.saturation_lookback,
        rel_delta=cfg.saturation_rel_delta):
    stop_reason = "weights saturated"
```

`VisionInsufficientTokensError` is caught around `agent.play_game()` and
breaks the loop on the spot — that's the third criterion.

## 5. Prompt economy

The system prompt is **constant** (~2 KB) — Vision sees it on every call
but the model's prompt cache will dedupe it. The per-turn message is
intentionally compact:

- one progress line
- one state line (state + grid hash + available actions)
- one colors line (top-8 histogram for timer / counter detection)
- one last_action line (with diff cells + transition + bbox)
- 0-3 warning lines (repetition / stuck / already-tried summary)
- the ASCII grid (~3 KB for a 64x64)
- one trailing instruction

Plus one image content block (~10 KB PNG base64). The model gets ~15 KB
per turn of *new* information, and CL weights cover the rest.

## 6. Tool-call retry behaviour

When Vision returns either no tool call or an unparseable one, the harness
replies with a corrective message and retries up to `MAX_INVALID_TOOL_RETRIES`
(2). After that the agent falls back to `state.available_actions[0]` and
logs `vision.fallback`. This guarantees the play loop makes progress even
under adversarial model output.

## 7. Safety / submission guardrails

- `ArcRunner.close_scorecard` in COMPETITION mode refuses to fire unless
  `ARC_ALLOW_LEADERBOARD_SUBMIT=yes-publish-to-leaderboard` is set.
- `vision-arc-compete` defaults to NOT closing the scorecard
  (`--publish=False`). It must be passed explicitly.
- The `submission/HOW_TO_SUBMIT.md` doc enumerates every step including the
  manual PR submission, so nothing about the leaderboard happens
  automatically.

## 8. Things to tune

If a particular game is hard:

- **Bigger image**: `--png-cell 12` doubles the PNG resolution.
- **Smaller per-turn budget**: `--max-turns-per-game 60` caps cost while
  you iterate.
- **Targeted retraining**: `--games <game_id>` runs only that game's passes.
- **Saturate slower**: `--saturation-rel-delta 0.005` makes the saturation
  detector less aggressive (so training will run longer).
