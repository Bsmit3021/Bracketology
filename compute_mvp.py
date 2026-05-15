"""Finals MVP candidate forecaster.

Scores every top-3 reg-season scorer on every still-alive playoff team by:

    score = (team_p_champion * 0.5)
          + (playoff_ppg / 30) * 0.3
          + (playoff_apg / 10) * 0.1
          + (playoff_rpg / 10) * 0.1

then normalizes across all candidates to produce an "MVP probability" share.
This is a formula-based proxy — not a learned model — useful as an "if the
Finals started today" leaderboard.

Inputs (must exist):
  - data/round_advancement.csv          (from predict_bracket.py)
  - data/nba_playoffs_boxscores_*.csv   (from nba_playoffs.py)
  - cached player stats (via build_features.fetch_player_stats)

Output:
  - data/mvp_candidates.csv

Run:
    python3 compute_mvp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_features import fetch_player_stats  # noqa: E402
from nba_api.stats.static import teams as _static_teams  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SEASON = "2025-26"

# Only candidates from teams with at least this much title shot are included.
MIN_TEAM_FINALS_PROB = 0.05


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def latest_boxscore_csv() -> Path | None:
    matches = sorted(DATA.glob("nba_playoffs_boxscores_*.csv"))
    return matches[-1] if matches else None


def main() -> int:
    adv_path = DATA / "round_advancement.csv"
    if not adv_path.exists():
        log(f"Missing {adv_path}. Run `python3 predict_bracket.py` first.")
        return 1
    adv = pd.read_csv(adv_path)

    box_path = latest_boxscore_csv()
    if box_path is None:
        log("No box scores CSV found. Run `python3 nba_playoffs.py` first.")
        return 1
    box = pd.read_csv(box_path)
    log(f"Loaded {len(box)} player-game rows from {box_path.name}")

    # Identify alive teams worth considering.
    alive = adv[adv["p_made_Finals"] > MIN_TEAM_FINALS_PROB].copy()
    if alive.empty:
        log("No teams with p_made_Finals > 5% — bracket may be wide-open or unset.")
        return 1
    log(f"Candidate teams ({len(alive)}): "
        + ", ".join(f"{r['team']}={r['p_champion']:.0%}"
                    for _, r in alive.iterrows()))

    # Get top-3 reg-season scorers per team (cached API call).
    player_stats = fetch_player_stats(SEASON)
    abbr_to_id = {t["abbreviation"]: t["id"] for t in _static_teams.get_teams()}
    id_to_abbr = {v: k for k, v in abbr_to_id.items()}

    candidates: list[dict] = []
    for _, team_row in alive.iterrows():
        team_abbr = team_row["team"]
        team_id = abbr_to_id.get(team_abbr)
        if team_id is None:
            continue
        team_players = player_stats[player_stats["TEAM_ID"] == team_id]
        top3 = team_players.nlargest(3, "PTS")[["PLAYER_ID", "PLAYER_NAME", "PTS"]]
        for _, p in top3.iterrows():
            name = p["PLAYER_NAME"]
            # Box score player_name field is "F. Last" (abbreviated initials)
            # whereas player_stats has "First Last". Match by last name + first
            # initial — robust enough for top scorers.
            parts = str(name).split()
            initial = parts[0][:1].upper() + "." if parts else ""
            last = " ".join(parts[1:]) if len(parts) > 1 else (parts[0] if parts else "")
            short_form = f"{initial} {last}"
            player_box = box[
                (box["team_abbr"] == team_abbr)
                & (box["player_name"].astype(str).str.strip() == short_form)
            ]
            # Fallback: try a substring match if the abbreviated form didn't hit.
            if player_box.empty:
                player_box = box[
                    (box["team_abbr"] == team_abbr)
                    & (box["player_name"].astype(str).str.contains(
                        last, case=False, na=False, regex=False))
                ]
            games_played = len(player_box)
            ppg = float(player_box["pts"].mean()) if games_played else 0.0
            apg = float(player_box["ast"].mean()) if games_played else 0.0
            rpg = float(player_box["reb"].mean()) if games_played else 0.0
            score = (
                team_row["p_champion"] * 0.50
                + (ppg / 30.0) * 0.30
                + (apg / 10.0) * 0.10
                + (rpg / 10.0) * 0.10
            )
            candidates.append({
                "player_name": name,
                "team_abbr": team_abbr,
                "p_team_champion": float(team_row["p_champion"]),
                "p_team_made_Finals": float(team_row["p_made_Finals"]),
                "playoff_games": games_played,
                "playoff_ppg": round(ppg, 1),
                "playoff_apg": round(apg, 1),
                "playoff_rpg": round(rpg, 1),
                "reg_season_ppg": round(float(p["PTS"]), 1),
                "mvp_score": float(score),
            })

    if not candidates:
        log("No MVP candidates derived. Check that box scores cover alive teams.")
        return 1

    df = pd.DataFrame(candidates).sort_values("mvp_score", ascending=False)
    total = df["mvp_score"].sum()
    df["mvp_prob"] = df["mvp_score"] / total if total > 0 else 0.0
    out = DATA / "mvp_candidates.csv"
    df.to_csv(out, index=False)

    log("\nTop 5 MVP candidates:")
    for _, c in df.head(5).iterrows():
        log(f"  {c['player_name']:<22s} {c['team_abbr']}  "
            f"mvp_prob={c['mvp_prob']:.1%}  ppg={c['playoff_ppg']:.1f}  "
            f"team_champ={c['p_team_champion']:.0%}")
    print(f"OK: wrote {len(df)} MVP candidates to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
