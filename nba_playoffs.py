"""
NBA 2025-26 Playoffs Scraper (Part A)

Produces two CSVs in the current working directory:
  - nba_playoffs_games_<date>.csv     (one row per playoff game)
  - nba_playoffs_boxscores_<date>.csv (one row per player per non-scheduled game)
"""

from __future__ import annotations

import csv
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from nba_api.stats.endpoints import (
    boxscoretraditionalv3,
    leaguedashplayerstats,
    leaguegamefinder,
    scheduleleaguev2,
)

SEASON = "2025-26"
LEAGUE_ID = "00"
ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
API_PAUSE_SEC = 0.6

OUTPUT_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GAMES_COLUMNS = [
    "game_id", "game_date_et", "status", "tipoff_et", "series", "game_label",
    "home_team", "away_team",
    "home_score", "away_score",
    "home_reg_season_leader", "home_reg_season_ppg",
    "away_reg_season_leader", "away_reg_season_ppg",
]

BOXSCORE_COLUMNS = [
    "game_id", "team_abbr", "player_name", "minutes",
    "pts", "reb", "ast", "stl", "blk", "pf", "tov",
    "fgm", "fga", "fg_pct", "three_pm", "three_pa", "three_pct",
    "ftm", "fta", "ft_pct", "plus_minus",
]


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# 1. Completed + in-progress playoff games (LeagueGameFinder)
# ---------------------------------------------------------------------------

def fetch_completed_and_inprogress_games() -> pd.DataFrame:
    log("Fetching completed/in-progress playoff games...")
    gf = leaguegamefinder.LeagueGameFinder(
        season_nullable=SEASON,
        season_type_nullable="Playoffs",
        league_id_nullable=LEAGUE_ID,
    )
    df = gf.get_data_frames()[0]
    if df.empty:
        return pd.DataFrame()

    # Each game has two rows (one per team). Split into home/away by matchup string.
    df["is_home"] = df["MATCHUP"].str.contains(" vs. ")
    home = df[df["is_home"]].copy()
    away = df[~df["is_home"]].copy()

    merged = home.merge(
        away,
        on="GAME_ID",
        suffixes=("_home", "_away"),
        how="inner",
    )

    rows = []
    for _, r in merged.iterrows():
        game_date = pd.to_datetime(r["GAME_DATE_home"]).date().isoformat()
        # LeagueGameFinder only returns games that have actually started, so all
        # rows here are "Final" or "In Progress". The endpoint doesn't expose a
        # status flag directly; we treat anything with both team PTS populated
        # and a WL recorded as Final. Otherwise mark In Progress.
        wl = r.get("WL_home")
        if pd.isna(wl) or wl in (None, ""):
            status = "In Progress"
        else:
            status = "Final"

        rows.append({
            "game_id": r["GAME_ID"],
            "game_date_et": game_date,
            "status": status,
            "tipoff_et": "",
            "series": "",
            "game_label": "",
            "home_team_id": int(r["TEAM_ID_home"]),
            "away_team_id": int(r["TEAM_ID_away"]),
            "home_team": r["TEAM_ABBREVIATION_home"],
            "away_team": r["TEAM_ABBREVIATION_away"],
            "home_score": int(r["PTS_home"]) if pd.notna(r["PTS_home"]) else "",
            "away_score": int(r["PTS_away"]) if pd.notna(r["PTS_away"]) else "",
        })

    out = pd.DataFrame(rows)
    log(f"  -> {len(out)} completed/in-progress games")
    return out


# ---------------------------------------------------------------------------
# 2. Scheduled playoff games (ScheduleLeagueV2)
# ---------------------------------------------------------------------------

def _extract_schedule_games(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """ScheduleLeagueV2 returns leagueSchedule.gameDates[].games[] (camelCase)."""
    league_sched = payload.get("leagueSchedule") or {}
    out: list[dict[str, Any]] = []
    for gd in league_sched.get("gameDates", []) or []:
        for g in gd.get("games", []) or []:
            out.append(g)
    return out


def _is_playoff_game(g: dict[str, Any]) -> bool:
    # game subtype/label fields vary by season; check several signals.
    sub = (g.get("gameSubtype") or "").lower()
    label = (g.get("gameLabel") or "").lower()
    sub_label = (g.get("gameSubLabel") or "").lower()
    week = (g.get("weekName") or "").lower()
    series = (g.get("seriesText") or "").lower()
    # In the past, weekName has been e.g. "Playoffs - Rd 1", and gameLabel like
    # "First Round" / "Conf. Semifinals" / "Conf. Finals" / "NBA Finals".
    hay = " ".join([sub, label, sub_label, week, series])
    if any(kw in hay for kw in (
        "playoff", "first round", "conf. semi", "semifinal",
        "conf. final", "conference final", "finals",
    )):
        # Exclude play-in tournament games which sometimes leak in.
        if "play-in" in hay or "play in" in hay:
            return False
        return True
    return False


def fetch_scheduled_playoff_games() -> pd.DataFrame:
    log("Fetching scheduled playoff games...")
    sched = scheduleleaguev2.ScheduleLeagueV2(season=SEASON, league_id=LEAGUE_ID)
    payload = sched.get_dict()
    games = _extract_schedule_games(payload)

    rows = []
    for g in games:
        if not _is_playoff_game(g):
            continue
        game_status = g.get("gameStatus")  # 1=Scheduled, 2=In Progress, 3=Final
        if game_status != 1:
            continue  # in-progress/final handled by LeagueGameFinder

        game_id = g.get("gameId") or ""
        # Tip-off time: gameDateTimeUTC -> ET
        dt_utc = g.get("gameDateTimeUTC")
        tipoff_et = ""
        game_date_et = ""
        if dt_utc:
            try:
                dt = datetime.fromisoformat(dt_utc.replace("Z", "+00:00"))
                dt_et = dt.astimezone(ET)
                tipoff_et = dt_et.strftime("%Y-%m-%d %H:%M ET")
                game_date_et = dt_et.date().isoformat()
            except ValueError:
                pass

        home = g.get("homeTeam") or {}
        away = g.get("awayTeam") or {}
        # Skip TBD/placeholder bracket slots (e.g. Finals games before participants are decided).
        if not (home.get("teamTricode") and away.get("teamTricode")):
            continue
        rows.append({
            "game_id": game_id,
            "game_date_et": game_date_et,
            "status": "Scheduled",
            "tipoff_et": tipoff_et,
            "series": g.get("seriesText") or "",
            "game_label": g.get("gameLabel") or "",
            "home_team_id": int(home.get("teamId") or 0),
            "away_team_id": int(away.get("teamId") or 0),
            "home_team": home.get("teamTricode") or "",
            "away_team": away.get("teamTricode") or "",
            "home_score": "",
            "away_score": "",
        })

    out = pd.DataFrame(rows)
    log(f"  -> {len(out)} scheduled playoff games")
    return out


# ---------------------------------------------------------------------------
# 3. Merge games
# ---------------------------------------------------------------------------

def merge_all_games(played: pd.DataFrame, scheduled: pd.DataFrame) -> pd.DataFrame:
    if played.empty and scheduled.empty:
        return pd.DataFrame()
    if played.empty:
        merged = scheduled
    elif scheduled.empty:
        merged = played
    else:
        merged = pd.concat([played, scheduled], ignore_index=True)
    merged = merged.drop_duplicates(subset=["game_id"], keep="first")
    merged = merged.sort_values(["game_date_et", "game_id"], na_position="last").reset_index(drop=True)
    return merged


# ---------------------------------------------------------------------------
# 4. Full box score per game
# ---------------------------------------------------------------------------

def _minutes_to_str(m: Any) -> str:
    if m is None or (isinstance(m, float) and pd.isna(m)):
        return ""
    s = str(m)
    # v3 format is sometimes "PT34M12.000S" or "34:12" or just a number; normalize.
    if s.startswith("PT"):
        try:
            body = s[2:]
            mm, _, rest = body.partition("M")
            ss, _, _ = rest.partition("S")
            return f"{int(float(mm))}:{int(float(ss or 0)):02d}"
        except (ValueError, TypeError):
            return s
    return s


def fetch_full_boxscore(game_id: str) -> list[dict[str, Any]]:
    try:
        bs = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
        data = bs.get_dict().get("boxScoreTraditional") or {}
    except Exception as exc:
        log(f"  ! boxscore fetch failed for {game_id}: {exc}")
        return []

    rows: list[dict[str, Any]] = []
    for side in ("homeTeam", "awayTeam"):
        team = data.get(side) or {}
        team_abbr = team.get("teamTricode") or ""
        for p in team.get("players") or []:
            stats = p.get("statistics") or {}
            full_name = p.get("nameI") or p.get("name") or (
                f"{p.get('firstName', '')} {p.get('familyName', '')}".strip()
            )
            rows.append({
                "game_id": game_id,
                "team_abbr": team_abbr,
                "player_name": full_name,
                "minutes": _minutes_to_str(stats.get("minutes") or stats.get("minutesCalculated")),
                "pts": stats.get("points"),
                "reb": stats.get("reboundsTotal"),
                "ast": stats.get("assists"),
                "stl": stats.get("steals"),
                "blk": stats.get("blocks"),
                "pf": stats.get("foulsPersonal"),
                "tov": stats.get("turnovers"),
                "fgm": stats.get("fieldGoalsMade"),
                "fga": stats.get("fieldGoalsAttempted"),
                "fg_pct": stats.get("fieldGoalsPercentage"),
                "three_pm": stats.get("threePointersMade"),
                "three_pa": stats.get("threePointersAttempted"),
                "three_pct": stats.get("threePointersPercentage"),
                "ftm": stats.get("freeThrowsMade"),
                "fta": stats.get("freeThrowsAttempted"),
                "ft_pct": stats.get("freeThrowsPercentage"),
                "plus_minus": stats.get("plusMinusPoints"),
            })
    return rows


# ---------------------------------------------------------------------------
# 5. Regular-season PPG leader per team
# ---------------------------------------------------------------------------

def fetch_regular_season_leaders() -> dict[int, tuple[str, float]]:
    log("Fetching regular-season PPG leaders by team...")
    ds = leaguedashplayerstats.LeagueDashPlayerStats(
        season=SEASON,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
    )
    df = ds.get_data_frames()[0]
    if df.empty:
        return {}
    # Best scorer per team by PPG.
    idx = df.groupby("TEAM_ID")["PTS"].idxmax()
    leaders = df.loc[idx, ["TEAM_ID", "PLAYER_NAME", "PTS"]]
    out: dict[int, tuple[str, float]] = {}
    for _, r in leaders.iterrows():
        out[int(r["TEAM_ID"])] = (str(r["PLAYER_NAME"]), float(r["PTS"]))
    log(f"  -> leaders for {len(out)} teams")
    return out


# ---------------------------------------------------------------------------
# 6/7. Build + write CSVs
# ---------------------------------------------------------------------------

def write_games_csv(games: pd.DataFrame, leaders: dict[int, tuple[str, float]], path: Path) -> int:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GAMES_COLUMNS)
        w.writeheader()
        for _, r in games.iterrows():
            h_lead = leaders.get(int(r["home_team_id"]), ("", ""))
            a_lead = leaders.get(int(r["away_team_id"]), ("", ""))
            w.writerow({
                "game_id": r["game_id"],
                "game_date_et": r["game_date_et"],
                "status": r["status"],
                "tipoff_et": r["tipoff_et"],
                "series": r["series"],
                "game_label": r["game_label"],
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "home_score": r["home_score"],
                "away_score": r["away_score"],
                "home_reg_season_leader": h_lead[0],
                "home_reg_season_ppg": h_lead[1],
                "away_reg_season_leader": a_lead[0],
                "away_reg_season_ppg": a_lead[1],
            })
    return len(games)


def write_boxscores_csv(games: pd.DataFrame, path: Path) -> int:
    non_scheduled = games[games["status"] != "Scheduled"]
    total_rows = 0
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BOXSCORE_COLUMNS)
        w.writeheader()
        for i, (_, r) in enumerate(non_scheduled.iterrows(), start=1):
            gid = str(r["game_id"])
            log(f"  [{i}/{len(non_scheduled)}] boxscore {gid} ({r['home_team']} vs {r['away_team']})")
            rows = fetch_full_boxscore(gid)
            for row in rows:
                w.writerow(row)
            total_rows += len(rows)
            time.sleep(API_PAUSE_SEC)
    return total_rows


def main() -> int:
    today = date.today().isoformat()
    games_path = OUTPUT_DIR / f"nba_playoffs_games_{today}.csv"
    box_path = OUTPUT_DIR / f"nba_playoffs_boxscores_{today}.csv"

    played = fetch_completed_and_inprogress_games()
    time.sleep(API_PAUSE_SEC)
    scheduled = fetch_scheduled_playoff_games()
    time.sleep(API_PAUSE_SEC)
    games = merge_all_games(played, scheduled)
    if games.empty:
        log("No playoff games found for season " + SEASON)
        return 1

    leaders = fetch_regular_season_leaders()
    time.sleep(API_PAUSE_SEC)

    n_games = write_games_csv(games, leaders, games_path)
    print(f"Wrote {n_games} rows to {games_path}")

    n_boxrows = write_boxscores_csv(games, box_path)
    print(f"Wrote {n_boxrows} rows to {box_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
