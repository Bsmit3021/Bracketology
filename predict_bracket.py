"""
Monte Carlo championship simulator (Part B3).

Loads Part A's games CSV (current bracket state), trained model from Part B2,
and current-season team advanced stats. Runs N simulations of the remaining
playoffs game-by-game, then emits:

    data/champion_probabilities.csv
    data/round_advancement.csv

Run:
    python3 predict_bracket.py [--sims 10000] [--seed 42]
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_features import (  # noqa: E402
    ADV_COLS,
    fetch_team_advanced,
    team_advanced_lookup,
)
from nba_api.stats.static import teams as _static_teams  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL_PATH = ROOT / "model.pkl"

SEASON = "2025-26"

# Bracket structure: pairings between rounds.
# R2 slot k draws from R1 slots in R1_TO_R2[k].
R1_TO_R2 = {0: (0, 3), 1: (1, 2), 2: (4, 7), 3: (5, 6)}
# R3 (CF) slot k draws from R2 slots in R2_TO_R3[k].
R2_TO_R3 = {0: (0, 1), 1: (2, 3)}
# R4 (Finals) slot 0 draws from R3 slots 0 and 1.
R3_TO_R4 = {0: (0, 1)}

# NBA 2-2-1-1-1 home schedule: higher seed at home for games 1, 2, 5, 7.
HIGHER_SEED_HOME_GAMES = {1, 2, 5, 7}


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------

def latest_games_csv() -> Path:
    matches = sorted(DATA.glob("nba_playoffs_games_*.csv"))
    if not matches:
        raise FileNotFoundError("No nba_playoffs_games_*.csv found in data/")
    return matches[-1]


def parse_game_id(gid: Any) -> tuple[int, int, int]:
    s = str(gid).zfill(10)
    return int(s[7]), int(s[8]), int(s[9])  # (round, slot, game_num)


def load_games() -> pd.DataFrame:
    path = latest_games_csv()
    log(f"Loading games CSV: {path.name}")
    df = pd.read_csv(path)
    before = len(df)
    df = df.dropna(subset=["home_team", "away_team"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        log(f"  dropped {dropped} placeholder rows with TBD teams")
    parsed = df["game_id"].apply(parse_game_id)
    df["round"] = [p[0] for p in parsed]
    df["slot"] = [p[1] for p in parsed]
    df["game_num"] = [p[2] for p in parsed]
    df["game_date"] = pd.to_datetime(df["game_date_et"], errors="coerce")
    return df


def load_model_bundle():
    bundle = joblib.load(MODEL_PATH)
    log(f"Loaded model: {bundle['model_name']} "
        f"(val log_loss={bundle['val_metrics']['log_loss']:.4f})")
    return bundle


def load_current_team_data():
    log(f"Loading current-season team advanced stats ({SEASON}, cached)...")
    df = fetch_team_advanced(SEASON)
    adv = team_advanced_lookup(df)
    abbr_to_id = {t["abbreviation"]: t["id"] for t in _static_teams.get_teams()}
    return adv, abbr_to_id


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def build_initial_state(games_df: pd.DataFrame) -> dict:
    """Return {(round, slot): SeriesState dict}.

    SeriesState:
      teams: (a_abbr, b_abbr) sorted alphabetically
      wins: {abbr: int}
      scheduled_home: {game_num: abbr}        — observed home assignments for unplayed games
      played_home:    {game_num: abbr}        — observed home assignments for completed games
      complete: bool
      winner: str | None
    """
    state: dict[tuple[int, int], dict] = {}

    completed = games_df[games_df["status"] == "Final"].sort_values("game_date")
    for _, r in completed.iterrows():
        key = (int(r["round"]), int(r["slot"]))
        teams = tuple(sorted([r["home_team"], r["away_team"]]))
        s = state.setdefault(key, {
            "teams": teams,
            "wins": {teams[0]: 0, teams[1]: 0},
            "scheduled_home": {},
            "played_home": {},
            "complete": False,
            "winner": None,
        })
        home_pts = float(r["home_score"])
        away_pts = float(r["away_score"])
        winner = r["home_team"] if home_pts > away_pts else r["away_team"]
        s["wins"][winner] += 1
        s["played_home"][int(r["game_num"])] = r["home_team"]

    scheduled = games_df[games_df["status"] == "Scheduled"]
    for _, r in scheduled.iterrows():
        key = (int(r["round"]), int(r["slot"]))
        teams = tuple(sorted([r["home_team"], r["away_team"]]))
        s = state.setdefault(key, {
            "teams": teams,
            "wins": {teams[0]: 0, teams[1]: 0},
            "scheduled_home": {},
            "played_home": {},
            "complete": False,
            "winner": None,
        })
        s["scheduled_home"][int(r["game_num"])] = r["home_team"]

    # Mark complete series.
    for s in state.values():
        a, b = s["teams"]
        if s["wins"][a] >= 4 or s["wins"][b] >= 4:
            s["complete"] = True
            s["winner"] = a if s["wins"][a] >= 4 else b

    return state


def build_team_rolling_form(games_df: pd.DataFrame) -> dict[str, dict]:
    """Compute last-5-playoff-game rolling form per team from completed games."""
    completed = games_df[games_df["status"] == "Final"].sort_values("game_date")
    history: dict[str, list[dict]] = {}
    for _, r in completed.iterrows():
        h, a = r["home_team"], r["away_team"]
        hp, ap = float(r["home_score"]), float(r["away_score"])
        history.setdefault(h, []).append({"pts": hp, "pa": ap, "won": int(hp > ap)})
        history.setdefault(a, []).append({"pts": ap, "pa": hp, "won": int(ap > hp)})

    rolling: dict[str, dict] = {}
    for team, hist in history.items():
        last5 = hist[-5:]
        pts = [h["pts"] for h in last5]
        pa = [h["pa"] for h in last5]
        wins = [h["won"] for h in last5]
        rolling[team] = {
            "pts": sum(pts) / len(pts),
            "pts_allowed": sum(pa) / len(pa),
            "margin": (sum(pts) - sum(pa)) / len(pts),
            "win_pct": sum(wins) / len(wins),
            "games": len(last5),
        }
    return rolling


# ---------------------------------------------------------------------------
# Home court for a hypothetical game
# ---------------------------------------------------------------------------

def home_team_for_game(
    series: dict,
    game_num: int,
    adv_lookup: dict[int, dict],
    abbr_to_id: dict[str, int],
) -> str:
    """Return the home team abbr for `game_num` in this series."""
    if game_num in series["scheduled_home"]:
        return series["scheduled_home"][game_num]
    if game_num in series["played_home"]:
        return series["played_home"][game_num]
    a, b = series["teams"]
    nra = adv_lookup.get(abbr_to_id.get(a, -1), {}).get("NET_RATING", 0.0)
    nrb = adv_lookup.get(abbr_to_id.get(b, -1), {}).get("NET_RATING", 0.0)
    higher = a if nra >= nrb else b
    lower = b if higher == a else a
    return higher if game_num in HIGHER_SEED_HOME_GAMES else lower


# ---------------------------------------------------------------------------
# Feature vector for a hypothetical game
# ---------------------------------------------------------------------------

EMPTY_ROLL = {"pts": np.nan, "pts_allowed": np.nan, "margin": np.nan,
              "win_pct": np.nan, "games": 0}


def build_feature_vector(
    team_abbr: str,
    opp_abbr: str,
    is_home: int,
    wins_team: int,
    wins_opp: int,
    rolling: dict[str, dict],
    adv_lookup: dict[int, dict],
    abbr_to_id: dict[str, int],
    feature_cols: list[str],
) -> np.ndarray:
    game_number = wins_team + wins_opp + 1
    team_id = abbr_to_id.get(team_abbr, -1)
    opp_id = abbr_to_id.get(opp_abbr, -1)
    rt = rolling.get(team_abbr, EMPTY_ROLL)
    ro = rolling.get(opp_abbr, EMPTY_ROLL)
    adv_t = adv_lookup.get(team_id, {})
    adv_o = adv_lookup.get(opp_id, {})

    features: dict[str, float] = {
        "is_home": float(is_home),
        "series_wins_team": float(wins_team),
        "series_wins_opp": float(wins_opp),
        "series_score_diff": float(wins_team - wins_opp),
        "series_game_number": float(game_number),
        "is_elimination_for_team": float(wins_opp == 3),
        "is_elimination_for_opp": float(wins_team == 3),
        # Rest: use constant 2 days both sides (rest_diff = 0). Small impact.
        "days_rest_team": 2.0,
        "days_rest_opp": 2.0,
        "rest_diff": 0.0,
        "roll_pts_team": rt["pts"],
        "roll_pts_allowed_team": rt["pts_allowed"],
        "roll_margin_team": rt["margin"],
        "roll_win_pct_team": rt["win_pct"],
        "roll_games_team": float(rt["games"]),
        "roll_pts_opp": ro["pts"],
        "roll_pts_allowed_opp": ro["pts_allowed"],
        "roll_margin_opp": ro["margin"],
        "roll_win_pct_opp": ro["win_pct"],
        "roll_games_opp": float(ro["games"]),
        # Assume healthy rosters — no live injury feed.
        "stars_active_team": 3.0,
        "stars_active_opp": 3.0,
    }
    for c in ADV_COLS:
        t = adv_t.get(c, np.nan)
        o = adv_o.get(c, np.nan)
        features[f"team_{c.lower()}"] = t
        features[f"opp_{c.lower()}"] = o
        features[f"diff_{c.lower()}"] = (t - o) if (not np.isnan(t) and not np.isnan(o)) else np.nan

    return np.array([features[c] for c in feature_cols], dtype=float).reshape(1, -1)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_series(
    series: dict,
    model,
    feature_cols: list[str],
    rolling: dict[str, dict],
    adv_lookup: dict[int, dict],
    abbr_to_id: dict[str, int],
    rng: random.Random,
) -> str:
    """Simulate the rest of a series; return the winner abbr.

    Mutates `series["wins"]` in place.
    """
    if series["complete"]:
        return series["winner"]
    a, b = series["teams"]
    while series["wins"][a] < 4 and series["wins"][b] < 4:
        game_num = series["wins"][a] + series["wins"][b] + 1
        home = home_team_for_game(series, game_num, adv_lookup, abbr_to_id)
        away = b if home == a else a
        feats = build_feature_vector(
            team_abbr=home, opp_abbr=away, is_home=1,
            wins_team=series["wins"][home], wins_opp=series["wins"][away],
            rolling=rolling, adv_lookup=adv_lookup, abbr_to_id=abbr_to_id,
            feature_cols=feature_cols,
        )
        p_home_wins = float(model.predict_proba(feats)[0, 1])
        winner = home if rng.random() < p_home_wins else away
        series["wins"][winner] += 1
    series["complete"] = True
    series["winner"] = a if series["wins"][a] >= 4 else b
    return series["winner"]


def simulate_bracket(
    initial_state: dict,
    model,
    feature_cols: list[str],
    rolling: dict[str, dict],
    adv_lookup: dict[int, dict],
    abbr_to_id: dict[str, int],
    rng: random.Random,
) -> dict:
    """Run one simulation forward from `initial_state`. Returns advancement dict."""
    state = copy.deepcopy(initial_state)
    advancement: dict[str, set[int]] = {}

    # Round 1 — credit anyone who appears in R1 with "reached R1" (trivially true).
    # Then simulate R1 if not done.
    for slot in range(8):
        s = state.get((1, slot))
        if not s:
            continue
        a, b = s["teams"]
        advancement.setdefault(a, set()).add(1)
        advancement.setdefault(b, set()).add(1)
        winner = simulate_series(s, model, feature_cols, rolling, adv_lookup, abbr_to_id, rng)
        advancement.setdefault(winner, set()).add(2)

    # Round 2 — instantiate any not-yet-seen R2 series from R1 winners.
    for r2_slot, (l_a, l_b) in R1_TO_R2.items():
        key = (2, r2_slot)
        if key in state:
            # Series exists in real data (current bracket).
            s = state[key]
        else:
            wa = state[(1, l_a)]["winner"]
            wb = state[(1, l_b)]["winner"]
            teams = tuple(sorted([wa, wb]))
            s = {
                "teams": teams,
                "wins": {teams[0]: 0, teams[1]: 0},
                "scheduled_home": {},
                "played_home": {},
                "complete": False,
                "winner": None,
            }
            state[key] = s
        winner = simulate_series(s, model, feature_cols, rolling, adv_lookup, abbr_to_id, rng)
        advancement.setdefault(winner, set()).add(3)

    # Round 3 (Conf Finals).
    for r3_slot, (l_a, l_b) in R2_TO_R3.items():
        key = (3, r3_slot)
        if key in state:
            s = state[key]
        else:
            wa = state[(2, l_a)]["winner"]
            wb = state[(2, l_b)]["winner"]
            teams = tuple(sorted([wa, wb]))
            s = {
                "teams": teams,
                "wins": {teams[0]: 0, teams[1]: 0},
                "scheduled_home": {},
                "played_home": {},
                "complete": False,
                "winner": None,
            }
            state[key] = s
        winner = simulate_series(s, model, feature_cols, rolling, adv_lookup, abbr_to_id, rng)
        advancement.setdefault(winner, set()).add(4)

    # Round 4 (Finals).
    key = (4, 0)
    if key in state:
        s = state[key]
    else:
        wa = state[(3, 0)]["winner"]
        wb = state[(3, 1)]["winner"]
        teams = tuple(sorted([wa, wb]))
        s = {
            "teams": teams,
            "wins": {teams[0]: 0, teams[1]: 0},
            "scheduled_home": {},
            "played_home": {},
            "complete": False,
            "winner": None,
        }
        state[key] = s
    champion = simulate_series(s, model, feature_cols, rolling, adv_lookup, abbr_to_id, rng)
    advancement.setdefault(champion, set()).add(5)
    return {"advancement": advancement, "champion": champion}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    games_df = load_games()
    bundle = load_model_bundle()
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    adv_lookup, abbr_to_id = load_current_team_data()

    initial_state = build_initial_state(games_df)
    rolling = build_team_rolling_form(games_df)

    # Sanity log of current bracket
    log("Current bracket state:")
    for (rnd, slot), s in sorted(initial_state.items()):
        a, b = s["teams"]
        log(f"  R{rnd} slot {slot}: {a} {s['wins'][a]}-{s['wins'][b]} {b}"
            f"{' [DONE: ' + s['winner'] + ']' if s['complete'] else ''}")

    # Collect the 16 playoff teams.
    all_teams: set[str] = set()
    for s in initial_state.values():
        all_teams.update(s["teams"])
    log(f"\nRunning {args.sims} sims (seed={args.seed}) over {len(all_teams)} teams...")

    rng = random.Random(args.seed)
    # round_counts[team][round] = number of sims they reached that round
    # round 1 = made playoffs (trivially 1), 2 = made R2, 3 = CF, 4 = Finals, 5 = Champion
    rounds = [1, 2, 3, 4, 5]
    counts: dict[str, dict[int, int]] = {t: {r: 0 for r in rounds} for t in all_teams}
    champion_counts: dict[str, int] = {t: 0 for t in all_teams}

    for i in range(args.sims):
        result = simulate_bracket(
            initial_state, model, feature_cols, rolling,
            adv_lookup, abbr_to_id, rng,
        )
        for team, reached in result["advancement"].items():
            for r in reached:
                if team in counts:
                    counts[team][r] = counts[team].get(r, 0) + 1
        champion_counts[result["champion"]] = champion_counts.get(result["champion"], 0) + 1

    # Build outputs.
    n = float(args.sims)
    champ_rows = [
        {"team": t, "champion_prob": champion_counts[t] / n}
        for t in all_teams
    ]
    champ_df = pd.DataFrame(champ_rows).sort_values("champion_prob", ascending=False)
    champ_path = DATA / "champion_probabilities.csv"
    champ_df.to_csv(champ_path, index=False)

    adv_rows = []
    for t in all_teams:
        adv_rows.append({
            "team": t,
            "p_made_R1": counts[t][1] / n,
            "p_made_R2": counts[t][2] / n,
            "p_made_CF": counts[t][3] / n,
            "p_made_Finals": counts[t][4] / n,
            "p_champion": counts[t][5] / n,
        })
    adv_df = pd.DataFrame(adv_rows).sort_values("p_champion", ascending=False)
    adv_path = DATA / "round_advancement.csv"
    adv_df.to_csv(adv_path, index=False)

    log(f"\nWrote {champ_path} and {adv_path}")
    log("Top 5 champion probabilities:")
    for _, r in champ_df.head(5).iterrows():
        log(f"  {r['team']}: {r['champion_prob']:.1%}")
    total = champ_df["champion_prob"].sum()
    log(f"\nProbability sum (should be 1.0): {total:.4f}")

    print(f"OK: {args.sims} sims, champion total prob {total:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
