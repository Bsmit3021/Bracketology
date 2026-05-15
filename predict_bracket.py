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
    fetch_player_stats,
    fetch_team_advanced,
    team_advanced_lookup,
)
from nba_api.stats.static import teams as _static_teams  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL_PATH = ROOT / "model.pkl"
SCORE_MODEL_PATH = ROOT / "score_model.pkl"

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


def load_score_model_bundle():
    """Return score-model bundle, or None if not trained yet."""
    if not SCORE_MODEL_PATH.exists():
        log("  (no score_model.pkl found; score predictions will be skipped)")
        return None
    bundle = joblib.load(SCORE_MODEL_PATH)
    log(f"Loaded score model: {bundle['model_name']} "
        f"(val MAE_avg={bundle['val_metrics']['mae_avg']:.2f} pts)")
    return bundle


def load_current_team_data():
    log(f"Loading current-season team advanced stats ({SEASON}, cached)...")
    df = fetch_team_advanced(SEASON)
    adv = team_advanced_lookup(df)
    abbr_to_id = {t["abbreviation"]: t["id"] for t in _static_teams.get_teams()}
    return adv, abbr_to_id


def _normalize_name(name: str) -> str:
    """Strip punctuation/whitespace for fuzzy player-name matching."""
    return "".join(c.lower() for c in str(name) if c.isalnum())


def build_stars_active_map(abbr_to_id: dict[str, int]) -> dict[str, int]:
    """Returns {team_abbr: stars_active_count (0-3)} based on top-3 reg-season
    scorers and the latest ESPN injury report. Falls back to 3 (all active)
    when injury data is missing.
    """
    id_to_abbr = {v: k for k, v in abbr_to_id.items()}
    # Top-3 scorers per team (cached call into build_features).
    try:
        player_stats = fetch_player_stats(SEASON)
    except Exception as exc:
        log(f"  ! player stats load failed, defaulting stars_active=3: {exc}")
        return {abbr: 3 for abbr in abbr_to_id}

    top3_by_abbr: dict[str, list[str]] = {}
    for team_id, grp in player_stats.groupby("TEAM_ID"):
        abbr = id_to_abbr.get(int(team_id))
        if not abbr:
            continue
        topn = grp.nlargest(3, "PTS")
        top3_by_abbr[abbr] = [str(n) for n in topn["PLAYER_NAME"].tolist()]

    # Load injuries — graceful fallback if unavailable / stale.
    inj_path = DATA / "injury_report.csv"
    if not inj_path.exists():
        log("  ! no injury_report.csv; defaulting stars_active=3 for all teams")
        return {abbr: 3 for abbr in top3_by_abbr}
    inj = pd.read_csv(inj_path)
    inj_out = inj[inj["status"] == "Out"]
    out_by_team: dict[str, set[str]] = {}
    for _, r in inj_out.iterrows():
        out_by_team.setdefault(r["team_abbr"], set()).add(_normalize_name(r["player_name"]))

    result: dict[str, int] = {}
    for abbr, names in top3_by_abbr.items():
        outs = out_by_team.get(abbr, set())
        active = sum(1 for n in names if _normalize_name(n) not in outs)
        result[abbr] = active
    # Anyone missing from player_stats (shouldn't happen) defaults to 3.
    for abbr in abbr_to_id:
        result.setdefault(abbr, 3)
    return result


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
    stars_active_map: dict[str, int] | None = None,
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
        # Live injury feed (ESPN) drives stars_active_*; falls back to 3 if no data.
        "stars_active_team": float(
            (stars_active_map or {}).get(team_abbr, 3)
        ),
        "stars_active_opp": float(
            (stars_active_map or {}).get(opp_abbr, 3)
        ),
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
    stars_active_map: dict[str, int] | None = None,
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
            feature_cols=feature_cols, stars_active_map=stars_active_map,
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
    stars_active_map: dict[str, int] | None = None,
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
        winner = simulate_series(s, model, feature_cols, rolling, adv_lookup, abbr_to_id, rng, stars_active_map)
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
        winner = simulate_series(s, model, feature_cols, rolling, adv_lookup, abbr_to_id, rng, stars_active_map)
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
        winner = simulate_series(s, model, feature_cols, rolling, adv_lookup, abbr_to_id, rng, stars_active_map)
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
    champion = simulate_series(s, model, feature_cols, rolling, adv_lookup, abbr_to_id, rng, stars_active_map)
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
    score_bundle = load_score_model_bundle()
    adv_lookup, abbr_to_id = load_current_team_data()

    initial_state = build_initial_state(games_df)
    rolling = build_team_rolling_form(games_df)
    stars_active_map = build_stars_active_map(abbr_to_id)
    knocked_out_teams = [k for k, v in sorted(stars_active_map.items()) if v < 3]
    if knocked_out_teams:
        log(f"Injury-adjusted teams: "
            + ", ".join(f"{k}={stars_active_map[k]}" for k in knocked_out_teams))

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
            adv_lookup, abbr_to_id, rng, stars_active_map,
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

    # Per-scheduled-game model predictions (consumed by Vegas comparison tab).
    upcoming_rows = []
    scheduled = games_df[games_df["status"] == "Scheduled"]
    for _, r in scheduled.iterrows():
        # Build a feature vector for the home team's perspective using the
        # *current* series state (from initial_state) and predict P(home wins).
        key = (int(r["round"]), int(r["slot"]))
        s = initial_state.get(key)
        if not s:
            continue
        home = r["home_team"]; away = r["away_team"]
        wins_home = s["wins"].get(home, 0)
        wins_away = s["wins"].get(away, 0)
        feats = build_feature_vector(
            team_abbr=home, opp_abbr=away, is_home=1,
            wins_team=wins_home, wins_opp=wins_away,
            rolling=rolling, adv_lookup=adv_lookup, abbr_to_id=abbr_to_id,
            feature_cols=feature_cols, stars_active_map=stars_active_map,
        )
        p_home = float(model.predict_proba(feats)[0, 1])

        pred_home_pts: float | None = None
        pred_away_pts: float | None = None
        if score_bundle is not None:
            score_feats = feats  # same column order
            score_pred = score_bundle["model"].predict(score_feats)
            # Targets: [pts_team, pts_opp] from the home perspective
            pred_home_pts = float(score_pred[0, 0])
            pred_away_pts = float(score_pred[0, 1])

        upcoming_rows.append({
            "game_id": str(r["game_id"]).zfill(10),
            "game_date_et": str(r.get("game_date_et", "")),
            "tipoff_et": r.get("tipoff_et", ""),
            "series": r.get("series", ""),
            "game_label": r.get("game_label", ""),
            "home_team": home,
            "away_team": away,
            "wins_home": wins_home,
            "wins_away": wins_away,
            "model_home_win_prob": p_home,
            "model_away_win_prob": 1.0 - p_home,
            "pred_home_pts": pred_home_pts,
            "pred_away_pts": pred_away_pts,
        })
    upcoming_df = pd.DataFrame(upcoming_rows)
    upcoming_path = DATA / "upcoming_game_predictions.csv"
    upcoming_df.to_csv(upcoming_path, index=False)
    log(f"Wrote {len(upcoming_df)} upcoming-game predictions to {upcoming_path}")

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
