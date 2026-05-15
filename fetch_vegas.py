"""Fetch Vegas moneyline odds for upcoming NBA playoff games.

Uses The Odds API (free tier: 500 requests/month at https://the-odds-api.com).

Setup:
    1. Sign up at https://the-odds-api.com for a free key.
    2. Set the environment variable:
         export ODDS_API_KEY="your-key-here"
       or drop a one-line file at config_odds.txt containing just the key.

If no key is found, this script exits 0 with a notice — the dashboard will
just show a "no odds available" empty state.

Writes data/vegas_lines.csv with one row per upcoming game keyed by team
abbreviations, including:
  - american odds for home & away
  - no-vig implied probabilities
  - bookmaker the line was sourced from

Run:
    python3 fetch_vegas.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)
CONFIG_FILE = ROOT / "config_odds.txt"

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"

# Map ESPN/Odds API team-display names to NBA tricodes used in our CSVs.
TEAM_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def get_api_key() -> str | None:
    key = os.environ.get("ODDS_API_KEY") or os.environ.get("THE_ODDS_API_KEY")
    if key:
        return key.strip()
    if CONFIG_FILE.exists():
        content = CONFIG_FILE.read_text().strip()
        if content and not content.startswith("#"):
            return content.splitlines()[0].strip()
    return None


def american_to_prob(odds: float) -> float:
    """Convert American moneyline to implied probability (with vig)."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 100.0 / (odds + 100.0)


def best_h2h_line(bookmakers: list[dict], home_name: str, away_name: str) -> dict | None:
    """Pick the first bookmaker that has a complete head-to-head market."""
    for bm in bookmakers:
        for market in bm.get("markets", []) or []:
            if market.get("key") != "h2h":
                continue
            outcomes = {o["name"]: o["price"]
                        for o in market.get("outcomes", []) or []
                        if "name" in o and "price" in o}
            if home_name in outcomes and away_name in outcomes:
                return {
                    "bookmaker": bm.get("title") or bm.get("key"),
                    "home_odds": float(outcomes[home_name]),
                    "away_odds": float(outcomes[away_name]),
                }
    return None


def fetch_odds(api_key: str) -> pd.DataFrame:
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    resp = requests.get(ODDS_API_URL, params=params, timeout=20)
    if resp.status_code == 401:
        raise RuntimeError("Odds API rejected the key (401). Verify ODDS_API_KEY.")
    if resp.status_code == 429:
        raise RuntimeError("Odds API rate-limited (429). You may be out of free credits.")
    resp.raise_for_status()
    games = resp.json()
    remaining = resp.headers.get("x-requests-remaining")
    if remaining is not None:
        log(f"  Odds API requests remaining this month: {remaining}")

    rows = []
    for g in games:
        home_name = g.get("home_team")
        away_name = g.get("away_team")
        home_abbr = TEAM_NAME_TO_ABBR.get(home_name)
        away_abbr = TEAM_NAME_TO_ABBR.get(away_name)
        if not (home_abbr and away_abbr):
            continue
        line = best_h2h_line(g.get("bookmakers", []) or [], home_name, away_name)
        if not line:
            continue
        h_implied = american_to_prob(line["home_odds"])
        a_implied = american_to_prob(line["away_odds"])
        # No-vig: normalize so probabilities sum to 1
        tot = h_implied + a_implied
        h_fair = h_implied / tot if tot else h_implied
        a_fair = a_implied / tot if tot else a_implied
        rows.append({
            "commence_time": g.get("commence_time"),
            "home_team": home_abbr,
            "away_team": away_abbr,
            "home_team_name": home_name,
            "away_team_name": away_name,
            "bookmaker": line["bookmaker"],
            "home_american_odds": line["home_odds"],
            "away_american_odds": line["away_odds"],
            "home_implied_prob": h_implied,
            "away_implied_prob": a_implied,
            "home_fair_prob": h_fair,
            "away_fair_prob": a_fair,
            "vig": tot - 1.0,
        })
    df = pd.DataFrame(rows)
    df["fetched_at"] = datetime.now(timezone.utc).isoformat()
    return df


def main() -> int:
    key = get_api_key()
    if not key:
        log("ODDS_API_KEY not set and no config_odds.txt found.")
        log("To enable Vegas lines, sign up at https://the-odds-api.com "
            "and run:  export ODDS_API_KEY=your-key-here")
        print("OK: no key — Vegas integration in dry mode")
        return 0
    try:
        df = fetch_odds(key)
    except Exception as exc:
        log(f"ERROR: Odds fetch failed: {exc}")
        return 1
    out = DATA / "vegas_lines.csv"
    df.to_csv(out, index=False)
    print(f"OK: wrote {len(df)} odds rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
