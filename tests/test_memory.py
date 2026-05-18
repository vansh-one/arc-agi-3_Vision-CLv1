"""Tests for the WeightsStore + saturation detector."""

from __future__ import annotations

from vision_arc_agi.memory import WeightsRecord, WeightsStore


def test_round_trip_record(tmp_path):
    store = WeightsStore(tmp_path)
    rec = WeightsRecord(blob="abc", cl_usage=123, note="t1")
    rec.stats.games_played = 5
    rec.stats.weight_size_history = [100, 110, 120]
    store.save(rec)
    loaded = store.load_latest()
    assert loaded is not None
    assert loaded.blob == "abc"
    assert loaded.cl_usage == 123
    assert loaded.stats.games_played == 5
    assert loaded.stats.weight_size_history == [100, 110, 120]


def test_saturation_detector():
    # Sizes grew → not saturated
    growing = [100, 200, 400, 800]
    assert not WeightsStore.is_saturated(growing, lookback=4, rel_delta=0.05)
    # Sizes stable → saturated
    stable = [1000, 1005, 998, 1002]
    assert WeightsStore.is_saturated(stable, lookback=4, rel_delta=0.05)
    # Too few samples → not saturated
    assert not WeightsStore.is_saturated([100], lookback=4)
