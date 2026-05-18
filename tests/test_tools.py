"""Tests for the tool-call parser."""

from __future__ import annotations

from vision_arc_agi.tools import TOOL_DEFS, parse_tool_call


def test_tool_defs_well_formed():
    names = {td["function"]["name"] for td in TOOL_DEFS}
    assert "take_action" in names
    assert "take_action_sequence" in names
    assert "analyze_with_python" in names
    assert "reset_level" in names
    assert "concede_level" in names


def test_parse_take_action_simple():
    d = parse_tool_call("take_action", {"action": 1, "reasoning": "go up"})
    assert d.kind == "action"
    assert d.action == 1


def test_parse_take_action_click():
    d = parse_tool_call("take_action", {"action": 6, "x": 12, "y": 34, "reasoning": "click"})
    assert d.kind == "action"
    assert d.action == 6 and d.x == 12 and d.y == 34


def test_parse_take_action_click_missing_xy():
    d = parse_tool_call("take_action", {"action": 6, "reasoning": "click but oops"})
    assert d.kind == "invalid"


def test_parse_take_action_out_of_range():
    d = parse_tool_call("take_action", {"action": 99, "reasoning": "x"})
    assert d.kind == "invalid"


def test_parse_sequence():
    d = parse_tool_call("take_action_sequence", {
        "actions": [1, 1, 4, 5], "reasoning": "go up twice then right then interact",
    })
    assert d.kind == "sequence"
    assert d.sequence == [1, 1, 4, 5]


def test_parse_sequence_too_short():
    d = parse_tool_call("take_action_sequence", {"actions": [1], "reasoning": "x"})
    assert d.kind == "invalid"


def test_parse_sequence_strips_clicks():
    d = parse_tool_call("take_action_sequence", {"actions": [1, 6, 4], "reasoning": "x"})
    # action 6 is stripped, leaving [1, 4]
    assert d.kind == "sequence"
    assert d.sequence == [1, 4]


def test_parse_analysis():
    d = parse_tool_call("analyze_with_python", {"code": "print(G.shape)"})
    assert d.kind == "analysis"
    assert "print" in (d.code or "")


def test_parse_unknown_name():
    d = parse_tool_call("foobar", {})
    assert d.kind == "invalid"
