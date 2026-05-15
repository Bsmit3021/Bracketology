"""Unit tests for Bracketology helpers.

These cover the pure-Python pieces of the pipeline — no network calls, no
trained model, no Streamlit. Run with: pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build_features import (  # noqa: E402
    ADV_COLS,
    _parse_minutes,
    team_advanced_lookup,
    top3_scorers_by_team,
)
from predict_bracket import (  # noqa: E402
    build_team_rolling_form,
    parse_game_id,
)


def test_parse_game_id_extracts_round_slot_and_game_num():
    # 0042500121 — R1 game between NYK and ATL, slot 2, game 1
    assert parse_game_id("0042500121") == (1, 2, 1)
    # 0042500236 — R2 SAS vs MIN, slot 3, game 6
    assert parse_game_id("0042500236") == (2, 3, 6)
    # Unpadded integer (pandas strips leading zeros) — must still parse correctly
    assert parse_game_id(42500407) == (4, 0, 7)


def test_parse_minutes_handles_string_and_edge_cases():
    assert _parse_minutes("34:12") == pytest.approx(34 + 12 / 60)
    assert _parse_minutes("0:00") == 0.0
    assert _parse_minutes("28") == 28.0  # plain numeric
    assert _parse_minutes(None) == 0.0
    assert _parse_minutes("") == 0.0
    assert _parse_minutes(float("nan")) == 0.0


def test_top3_scorers_returns_top_three_per_team():
    df = pd.DataFrame([
        {"TEAM_ID": 1, "PLAYER_ID": 10, "PTS": 30.0},
        {"TEAM_ID": 1, "PLAYER_ID": 11, "PTS": 25.0},
        {"TEAM_ID": 1, "PLAYER_ID": 12, "PTS": 20.0},
        {"TEAM_ID": 1, "PLAYER_ID": 13, "PTS": 15.0},  # excluded
        {"TEAM_ID": 2, "PLAYER_ID": 20, "PTS": 28.0},
        {"TEAM_ID": 2, "PLAYER_ID": 21, "PTS": 14.0},
    ])
    result = top3_scorers_by_team(df)
    assert set(result[1]) == {10, 11, 12}
    assert set(result[2]) == {20, 21}
    # Order check: highest PTS comes first
    assert result[1][0] == 10


def test_team_advanced_lookup_has_all_adv_cols_per_team():
    rows = [
        {"TEAM_ID": 100, **{c: float(i) for i, c in enumerate(ADV_COLS)}},
        {"TEAM_ID": 200, **{c: float(i + 10) for i, c in enumerate(ADV_COLS)}},
    ]
    df = pd.DataFrame(rows)
    out = team_advanced_lookup(df)
    assert set(out.keys()) == {100, 200}
    for tid in out:
        assert set(out[tid].keys()) == set(ADV_COLS)
    # First ADV_COL is OFF_RATING — values should round-trip
    assert out[100]["OFF_RATING"] == 0.0
    assert out[200]["OFF_RATING"] == 10.0


def test_rolling_form_averages_match_inputs_and_are_symmetric():
    """Two-game toy schedule between AAA and BBB — rolling avgs must be exact."""
    df = pd.DataFrame([
        {"home_team": "AAA", "away_team": "BBB", "home_score": 100, "away_score": 90,
         "status": "Final", "game_date": pd.Timestamp("2024-01-01")},
        {"home_team": "BBB", "away_team": "AAA", "home_score": 110, "away_score": 105,
         "status": "Final", "game_date": pd.Timestamp("2024-01-03")},
    ])
    out = build_team_rolling_form(df)

    # AAA scored 100 then 105 → avg 102.5; allowed 90 then 110 → avg 100.0
    assert out["AAA"]["games"] == 2
    assert out["AAA"]["pts"] == pytest.approx(102.5)
    assert out["AAA"]["pts_allowed"] == pytest.approx(100.0)
    assert out["AAA"]["margin"] == pytest.approx(2.5)
    assert out["AAA"]["win_pct"] == pytest.approx(0.5)

    # BBB symmetric: 90 then 110 scored → 100; 100 then 105 allowed → 102.5
    assert out["BBB"]["pts"] == pytest.approx(100.0)
    assert out["BBB"]["pts_allowed"] == pytest.approx(102.5)
    assert out["BBB"]["margin"] == pytest.approx(-2.5)
    assert out["BBB"]["win_pct"] == pytest.approx(0.5)
