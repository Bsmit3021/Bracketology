"""Fetch NBA injury status from ESPN's public JSON API.

Writes data/injury_report.csv with one row per (team, player) injury entry.
Used by predict_bracket.py to set `stars_active` dynamically per team instead
of hard-coding 3.

ESPN endpoint is unauthenticated and stable. If unreachable, this script
exits non-zero and downstream code falls back to the previous default.

Run:
    python3 fetch_injuries.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

ESPN_INJURIES_URL = (
    "https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
)
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def fetch_espn_injuries() -> pd.DataFrame:
    """Returns DataFrame with columns: team_abbr, team_name, player_name,
    status, type, date, comment, fetched_at."""
    resp = requests.get(
        ESPN_INJURIES_URL,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=20,
    )
    resp.raise_for_status()
    payload = resp.json()
    rows: list[dict] = []
    for team in payload.get("injuries", []) or []:
        team_name = team.get("displayName") or ""
        for inj in team.get("injuries", []) or []:
            athlete = inj.get("athlete") or {}
            team_obj = athlete.get("team") or {}
            rows.append({
                "team_abbr": team_obj.get("abbreviation") or "",
                "team_name": team_name,
                "player_name": athlete.get("displayName") or "",
                "status": inj.get("status") or "",
                "type": inj.get("type") or "",
                "date": inj.get("date") or "",
                "comment": inj.get("shortComment") or "",
            })
    df = pd.DataFrame(rows)
    df["fetched_at"] = datetime.now(timezone.utc).isoformat()
    return df


def main() -> int:
    try:
        df = fetch_espn_injuries()
    except Exception as exc:
        log(f"ERROR: ESPN injuries fetch failed: {exc}")
        return 1
    out = DATA / "injury_report.csv"
    df.to_csv(out, index=False)
    n_out = int((df["status"] == "Out").sum())
    n_dtd = int((df["status"] == "Day-To-Day").sum())
    print(f"OK: wrote {len(df)} injury rows to {out} "
          f"({n_out} Out, {n_dtd} Day-To-Day)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
