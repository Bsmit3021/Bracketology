"""
Historical playoff features (Part B1).

For each season in SEASONS, builds one row per (game, team perspective) — two
rows per game — labeled `won` (1/0). Outputs data/historical_playoff_features.csv.

API responses are cached under ./cache/ so re-runs are fast.

Run:
    python3 build_features.py                 # all seasons
    python3 build_features.py --season 2023-24 # just one (smoke test)
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nba_api.stats.endpoints import (
    boxscoretraditionalv2,
    leaguedashplayerstats,
    leaguedashteamstats,
    leaguegamefinder,
)

LEAGUE_ID = "00"
API_PAUSE_SEC = 0.6

# 2019-20 bubble skipped per plan.
DEFAULT_SEASONS = [
    "2017-18", "2018-19", "2020-21", "2021-22",
    "2022-23", "2023-24", "2024-25",
]

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache"
DATA_DIR = ROOT / "data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Generic cache
# ---------------------------------------------------------------------------

def _cache_path(name: str, params: dict[str, Any]) -> Path:
    keystr = name + "|" + "|".join(f"{k}={params[k]}" for k in sorted(params))
    h = hashlib.md5(keystr.encode()).hexdigest()
    return CACHE_DIR / f"{name}_{h}.pkl"


def cached_fetch(name: str, params: dict[str, Any], fetch_fn) -> Any:
    p = _cache_path(name, params)
    if p.exists():
        with p.open("rb") as f:
            return pickle.load(f)
    time.sleep(API_PAUSE_SEC)
    result = fetch_fn()
    with p.open("wb") as f:
        pickle.dump(result, f)
    return result


# ---------------------------------------------------------------------------
# Per-season fetches
# ---------------------------------------------------------------------------

def fetch_playoff_games(season: str) -> pd.DataFrame:
    def _do():
        gf = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            season_type_nullable="Playoffs",
            league_id_nullable=LEAGUE_ID,
        )
        return gf.get_data_frames()[0]
    return cached_fetch("playoff_games", {"season": season}, _do)


def fetch_team_advanced(season: str) -> pd.DataFrame:
    def _do():
        ds = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="Per100Possessions",
        )
        return ds.get_data_frames()[0]
    return cached_fetch("team_advanced", {"season": season}, _do)


def fetch_player_stats(season: str) -> pd.DataFrame:
    def _do():
        ds = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_detailed="PerGame",
        )
        return ds.get_data_frames()[0]
    return cached_fetch("player_stats", {"season": season}, _do)


def fetch_boxscore_v2(game_id: str) -> pd.DataFrame:
    def _do():
        bs = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        return bs.get_data_frames()[0]  # player stats df
    return cached_fetch("box_v2", {"game_id": game_id}, _do)


# ---------------------------------------------------------------------------
# Derived: top-3 reg-season scorers per team, advanced stats lookup
# ---------------------------------------------------------------------------

def top3_scorers_by_team(player_stats: pd.DataFrame) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for team_id, grp in player_stats.groupby("TEAM_ID"):
        topn = grp.nlargest(3, "PTS")
        out[int(team_id)] = [int(pid) for pid in topn["PLAYER_ID"].tolist()]
    return out


ADV_COLS = ["OFF_RATING", "DEF_RATING", "NET_RATING", "PACE",
            "TS_PCT", "AST_PCT", "OREB_PCT", "DREB_PCT", "TM_TOV_PCT"]


def team_advanced_lookup(team_adv: pd.DataFrame) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for _, r in team_adv.iterrows():
        out[int(r["TEAM_ID"])] = {c: float(r[c]) if pd.notna(r[c]) else float("nan") for c in ADV_COLS}
    return out


# ---------------------------------------------------------------------------
# Box-score minutes parsing for stars_active_count
# ---------------------------------------------------------------------------

def _parse_minutes(m: Any) -> float:
    if m is None or (isinstance(m, float) and pd.isna(m)):
        return 0.0
    s = str(m).strip()
    if not s:
        return 0.0
    if ":" in s:
        try:
            mm, ss = s.split(":")
            return float(mm) + float(ss) / 60.0
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def stars_active_for_team(game_id: str, team_id: int, star_player_ids: list[int]) -> int:
    if not star_player_ids:
        return 0
    box = fetch_boxscore_v2(game_id)
    sub = box[(box["TEAM_ID"] == team_id) & (box["PLAYER_ID"].isin(star_player_ids))]
    count = 0
    for _, row in sub.iterrows():
        if _parse_minutes(row.get("MIN")) >= 10.0:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Per-season feature building
# ---------------------------------------------------------------------------

def build_season_features(season: str, include_stars: bool = True) -> pd.DataFrame:
    log(f"=== Season {season} ===")
    raw = fetch_playoff_games(season)
    if raw.empty:
        log(f"  no playoff games for {season}")
        return pd.DataFrame()

    raw = raw.copy()
    raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"])
    raw["is_home"] = raw["MATCHUP"].str.contains(" vs. ")
    raw["WON"] = (raw["WL"] == "W").astype(int)

    # Build per-game pivot (home/away) to identify matchups + opponent IDs.
    home = raw[raw["is_home"]].copy()
    away = raw[~raw["is_home"]].copy()
    merged = home.merge(
        away,
        on="GAME_ID",
        suffixes=("_h", "_a"),
        how="inner",
    )
    # Map game_id -> matchup metadata.
    game_meta: dict[str, dict[str, Any]] = {}
    for _, r in merged.iterrows():
        game_meta[r["GAME_ID"]] = {
            "date": r["GAME_DATE_h"],
            "home_team_id": int(r["TEAM_ID_h"]),
            "away_team_id": int(r["TEAM_ID_a"]),
            "home_pts": int(r["PTS_h"]) if pd.notna(r["PTS_h"]) else None,
            "away_pts": int(r["PTS_a"]) if pd.notna(r["PTS_a"]) else None,
        }

    # Series ID = sorted pair of team IDs.
    def series_key(t1: int, t2: int) -> tuple[int, int]:
        return (min(t1, t2), max(t1, t2))

    # Order games chronologically for series-context / rolling computations.
    raw_sorted = raw.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    team_adv = fetch_team_advanced(season)
    adv_lookup = team_advanced_lookup(team_adv)

    player_stats = fetch_player_stats(season)
    stars_by_team = top3_scorers_by_team(player_stats)

    # Trackers — series wins so far, last game date per team, rolling history per team.
    series_wins: dict[tuple[int, int], dict[int, int]] = {}
    last_game_date: dict[int, datetime] = {}
    playoff_history: dict[int, list[dict[str, Any]]] = {}

    rows: list[dict[str, Any]] = []
    games_seen: set[str] = set()
    # Iterate by GAME_ID in date order. raw_sorted has 2 rows per game; process
    # each game once when we see its first row.
    for _, r in raw_sorted.iterrows():
        gid = r["GAME_ID"]
        if gid in games_seen:
            continue
        games_seen.add(gid)
        meta = game_meta.get(gid)
        if not meta:
            continue

        home_id = meta["home_team_id"]
        away_id = meta["away_team_id"]
        date = meta["date"]
        home_pts = meta["home_pts"]
        away_pts = meta["away_pts"]
        if home_pts is None or away_pts is None:
            continue  # incomplete game

        skey = series_key(home_id, away_id)
        wins_so_far = series_wins.setdefault(skey, {home_id: 0, away_id: 0})
        # Ensure both keys present (in case of an edge case).
        wins_so_far.setdefault(home_id, 0)
        wins_so_far.setdefault(away_id, 0)
        wins_before_home = wins_so_far[home_id]
        wins_before_away = wins_so_far[away_id]
        game_number = wins_before_home + wins_before_away + 1

        # Rest days
        rest_home = (date - last_game_date[home_id]).days if home_id in last_game_date else None
        rest_away = (date - last_game_date[away_id]).days if away_id in last_game_date else None

        # Rolling form over last 5 playoff games (from history before this game)
        def rolling_form(team_id: int) -> dict[str, float]:
            hist = playoff_history.get(team_id, [])[-5:]
            if not hist:
                return {"pts": float("nan"), "pts_allowed": float("nan"),
                        "margin": float("nan"), "win_pct": float("nan"),
                        "games": 0}
            pts = [h["pts"] for h in hist]
            pa = [h["pa"] for h in hist]
            wins = [h["won"] for h in hist]
            return {
                "pts": sum(pts) / len(pts),
                "pts_allowed": sum(pa) / len(pa),
                "margin": (sum(pts) - sum(pa)) / len(pts),
                "win_pct": sum(wins) / len(wins),
                "games": len(hist),
            }

        home_form = rolling_form(home_id)
        away_form = rolling_form(away_id)

        # Stars active (expensive — boxscore fetch). Cached so re-runs are fast.
        if include_stars:
            home_stars = stars_active_for_team(gid, home_id, stars_by_team.get(home_id, []))
            away_stars = stars_active_for_team(gid, away_id, stars_by_team.get(away_id, []))
        else:
            home_stars = away_stars = -1

        # Emit one row per perspective.
        for perspective in ("home", "away"):
            if perspective == "home":
                team_id, opp_id = home_id, away_id
                is_home = 1
                wins_before_team, wins_before_opp = wins_before_home, wins_before_away
                rest_team, rest_opp = rest_home, rest_away
                form_team, form_opp = home_form, away_form
                stars_team, stars_opp = home_stars, away_stars
                won = 1 if home_pts > away_pts else 0
                pts_team, pts_opp = home_pts, away_pts
            else:
                team_id, opp_id = away_id, home_id
                is_home = 0
                wins_before_team, wins_before_opp = wins_before_away, wins_before_home
                rest_team, rest_opp = rest_away, rest_home
                form_team, form_opp = away_form, home_form
                stars_team, stars_opp = away_stars, home_stars
                won = 1 if away_pts > home_pts else 0
                pts_team, pts_opp = away_pts, home_pts

            adv_team = adv_lookup.get(team_id, {})
            adv_opp = adv_lookup.get(opp_id, {})

            row = {
                "season": season,
                "game_id": gid,
                "game_date": date.date().isoformat(),
                "team_id": team_id,
                "opp_team_id": opp_id,
                "is_home": is_home,
                "won": won,
                # Regression targets for the score-prediction model
                "pts_team": pts_team,
                "pts_opp": pts_opp,
                # Series context
                "series_wins_team": wins_before_team,
                "series_wins_opp": wins_before_opp,
                "series_score_diff": wins_before_team - wins_before_opp,
                "series_game_number": game_number,
                "is_elimination_for_team": int(wins_before_opp == 3),
                "is_elimination_for_opp": int(wins_before_team == 3),
                # Rest
                "days_rest_team": rest_team if rest_team is not None else "",
                "days_rest_opp": rest_opp if rest_opp is not None else "",
                "rest_diff": (rest_team - rest_opp) if (rest_team is not None and rest_opp is not None) else "",
                # Rolling form (last 5 playoff games)
                **{f"roll_{k}_team": v for k, v in form_team.items()},
                **{f"roll_{k}_opp": v for k, v in form_opp.items()},
                # Stars
                "stars_active_team": stars_team,
                "stars_active_opp": stars_opp,
            }
            # Team advanced stats (team + opp + diffs)
            for c in ADV_COLS:
                t = adv_team.get(c, float("nan"))
                o = adv_opp.get(c, float("nan"))
                row[f"team_{c.lower()}"] = t
                row[f"opp_{c.lower()}"] = o
                row[f"diff_{c.lower()}"] = (t - o) if pd.notna(t) and pd.notna(o) else float("nan")
            rows.append(row)

        # Update trackers AFTER emitting the row for this game.
        home_won = home_pts > away_pts
        if home_won:
            wins_so_far[home_id] += 1
        else:
            wins_so_far[away_id] += 1
        last_game_date[home_id] = date
        last_game_date[away_id] = date
        playoff_history.setdefault(home_id, []).append(
            {"pts": home_pts, "pa": away_pts, "won": int(home_won)}
        )
        playoff_history.setdefault(away_id, []).append(
            {"pts": away_pts, "pa": home_pts, "won": int(not home_won)}
        )

    df = pd.DataFrame(rows)
    log(f"  -> {len(df)} feature rows ({len(games_seen)} games)")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", help="Build features for a single season (smoke test)")
    ap.add_argument("--no-stars", action="store_true",
                    help="Skip stars_active feature (skips per-game boxscore calls)")
    ap.add_argument("--output", default=str(DATA_DIR / "historical_playoff_features.csv"))
    args = ap.parse_args()

    seasons = [args.season] if args.season else DEFAULT_SEASONS

    frames: list[pd.DataFrame] = []
    for s in seasons:
        df = build_season_features(s, include_stars=not args.no_stars)
        if not df.empty:
            frames.append(df)

    if not frames:
        log("No data built.")
        return 1
    out = pd.concat(frames, ignore_index=True)
    out_path = Path(args.output)
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
