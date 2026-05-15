"""
NBA Playoffs 2025-26 Predictor — Streamlit dashboard.

Run:
    streamlit run dashboard.py
"""

from __future__ import annotations

import datetime
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL_PATH = ROOT / "model.pkl"

st.set_page_config(
    page_title="NBA Playoffs 2025-26 Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# NBA team identity — primary colors + full names
# ---------------------------------------------------------------------------

TEAM_COLORS = {
    "ATL": "#E03A3E", "BOS": "#007A33", "BKN": "#000000", "CHA": "#1D1160",
    "CHI": "#CE1141", "CLE": "#860038", "DAL": "#00538C", "DEN": "#0E2240",
    "DET": "#1D42BA", "GSW": "#1D428A", "HOU": "#CE1141", "IND": "#FDBB30",
    "LAC": "#C8102E", "LAL": "#552583", "MEM": "#5D76A9", "MIA": "#98002E",
    "MIL": "#00471B", "MIN": "#236192", "NOP": "#85714D", "NYK": "#F58426",
    "OKC": "#007AC1", "ORL": "#0077C0", "PHI": "#006BB6", "PHX": "#E56020",
    "POR": "#E03A3E", "SAC": "#5A2D81", "SAS": "#000000", "TOR": "#CE1141",
    "UTA": "#002B5C", "WAS": "#002B5C",
}

TEAM_NAMES = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs", "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}


def team_color(abbr: str) -> str:
    return TEAM_COLORS.get(abbr, "#666666")


def team_name(abbr: str) -> str:
    return TEAM_NAMES.get(abbr, abbr)


# ---------------------------------------------------------------------------
# Global CSS — dark theme with NBA accents
# ---------------------------------------------------------------------------

CSS = """
<style>
.stApp {
    background: linear-gradient(180deg, #0E1117 0%, #131720 100%);
}
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }
h1, h2, h3 { letter-spacing: -0.02em; }
.hero-title {
    font-size: 2.4rem; font-weight: 800; line-height: 1.05;
    background: linear-gradient(90deg, #FB8332 0%, #FFD180 60%, #FFF 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    color: #B0B7C3; font-size: 0.95rem; margin-top: -4px; margin-bottom: 0.5rem;
}
.contender-card {
    border-radius: 14px;
    padding: 18px 22px;
    background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    color: white;
    height: 130px;
    position: relative;
    overflow: hidden;
}
.contender-card::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 4px;
    background: var(--accent);
}
.contender-rank {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
    color: #888; text-transform: uppercase;
}
.contender-team {
    font-size: 1.9rem; font-weight: 800; margin-top: 4px;
    color: white; line-height: 1.0;
}
.contender-name {
    font-size: 0.78rem; color: #A0A6B3; margin-top: 2px;
}
.contender-prob {
    position: absolute; right: 18px; bottom: 14px;
    font-size: 2.2rem; font-weight: 800; color: var(--accent);
}
.contender-prob-label {
    position: absolute; right: 18px; bottom: 8px;
    font-size: 0.65rem; text-transform: uppercase; color: #6B7280; letter-spacing: 0.1em;
}
.section-title {
    font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.15em;
    color: #8B95A7; font-weight: 700; margin-bottom: 6px; margin-top: 6px;
}
.series-card {
    border-radius: 12px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    padding: 12px 14px;
    margin-bottom: 8px;
}
.series-card.done { opacity: 0.55; }
.series-row {
    display: flex; justify-content: space-between; align-items: center;
    margin: 4px 0; font-size: 0.95rem; color: white;
}
.series-team { font-weight: 700; }
.series-score { font-variant-numeric: tabular-nums; color: #C8CDD7; font-weight: 700; }
.series-meta {
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;
    color: #6B7280; margin-bottom: 6px;
}
.kpi-tag {
    display: inline-block; padding: 2px 8px; border-radius: 10px;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.06em;
    background: rgba(251, 131, 50, 0.15); color: #FB8332;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: rgba(255,255,255,0.02); padding: 4px;
    border-radius: 10px; border: 1px solid rgba(255,255,255,0.06);
}
.stTabs [data-baseweb="tab"] {
    height: 36px; padding: 0 16px; border-radius: 6px; color: #B0B7C3;
}
.stTabs [aria-selected="true"] {
    background: rgba(251, 131, 50, 0.18) !important; color: #FB8332 !important;
}
div[data-testid="stMetricLabel"] { font-size: 0.72rem; color: #8B95A7; text-transform: uppercase; letter-spacing: 0.1em; }
div[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 800; color: white; }
.dataframe { border-radius: 8px; overflow: hidden; }
hr { border-color: rgba(255,255,255,0.08); margin: 12px 0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def latest_csv(pattern: str) -> Path | None:
    matches = sorted(DATA.glob(pattern))
    return matches[-1] if matches else None


@st.cache_data(ttl=600)
def load_data():
    games_path = latest_csv("nba_playoffs_games_*.csv")
    box_path = latest_csv("nba_playoffs_boxscores_*.csv")
    champ_path = DATA / "champion_probabilities.csv"
    adv_path = DATA / "round_advancement.csv"

    games = pd.read_csv(games_path) if games_path else pd.DataFrame()
    if not games.empty:
        games = games.dropna(subset=["home_team", "away_team"]).reset_index(drop=True)
        games["game_id_str"] = games["game_id"].astype(str).str.zfill(10)
        games["round"] = games["game_id_str"].str[7].astype(int)
        games["slot"] = games["game_id_str"].str[8].astype(int)
        games["game_num"] = games["game_id_str"].str[9].astype(int)
        games["game_date_et"] = pd.to_datetime(games["game_date_et"], errors="coerce")
    else:
        # Empty schema so downstream filters don't crash on missing columns.
        empty_cols = [
            "game_id", "game_id_str", "game_date_et", "status", "tipoff_et",
            "series", "game_label", "home_team", "away_team",
            "home_score", "away_score",
            "home_reg_season_leader", "home_reg_season_ppg",
            "away_reg_season_leader", "away_reg_season_ppg",
            "round", "slot", "game_num",
        ]
        games = pd.DataFrame({c: pd.Series(dtype="object") for c in empty_cols})
        games["game_date_et"] = pd.to_datetime(games["game_date_et"], errors="coerce")

    boxscores = pd.read_csv(box_path) if box_path else pd.DataFrame()
    if not boxscores.empty:
        boxscores["game_id_str"] = boxscores["game_id"].astype(str).str.zfill(10)

    champ = pd.read_csv(champ_path) if champ_path.exists() else pd.DataFrame()
    adv = pd.read_csv(adv_path) if adv_path.exists() else pd.DataFrame()

    bt_results = pd.read_csv(DATA / "backtest_results.csv") if (DATA / "backtest_results.csv").exists() else pd.DataFrame()
    bt_round = pd.read_csv(DATA / "backtest_round_brier.csv") if (DATA / "backtest_round_brier.csv").exists() else pd.DataFrame()
    bt_calib = pd.read_csv(DATA / "backtest_calibration.csv") if (DATA / "backtest_calibration.csv").exists() else pd.DataFrame()
    injuries = pd.read_csv(DATA / "injury_report.csv") if (DATA / "injury_report.csv").exists() else pd.DataFrame()
    vegas = pd.read_csv(DATA / "vegas_lines.csv") if (DATA / "vegas_lines.csv").exists() else pd.DataFrame()
    upcoming = pd.read_csv(DATA / "upcoming_game_predictions.csv") if (DATA / "upcoming_game_predictions.csv").exists() else pd.DataFrame()

    bundle = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

    return {
        "games": games,
        "boxscores": boxscores,
        "champ": champ,
        "adv": adv,
        "bt_results": bt_results,
        "bt_round": bt_round,
        "bt_calib": bt_calib,
        "injuries": injuries,
        "vegas": vegas,
        "upcoming": upcoming,
        "model": bundle,
        "games_path": games_path,
        "box_path": box_path,
    }


def run_pipeline(steps: list[str]) -> tuple[bool, str]:
    """Run scraper and/or predictor as a subprocess. Returns (ok, log)."""
    log_lines: list[str] = []
    for step in steps:
        log_lines.append(f"$ python3 {step}.py")
        try:
            proc = subprocess.run(
                [sys.executable, str(ROOT / f"{step}.py")],
                cwd=str(ROOT),
                check=False,
                capture_output=True,
                text=True,
                timeout=900,
            )
            log_lines.append(proc.stdout.strip())
            if proc.stderr.strip():
                log_lines.append(proc.stderr.strip())
            if proc.returncode != 0:
                return False, "\n".join(log_lines)
        except Exception as exc:
            log_lines.append(f"ERROR: {exc}")
            return False, "\n".join(log_lines)
    return True, "\n".join(log_lines)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🏀 Playoffs Predictor")
    st.caption("Live championship forecast powered by a model trained on 7 seasons of historical playoff data.")
    st.divider()

    st.markdown("**Refresh data**")
    col_a, col_b, col_c = st.columns(3)
    if col_a.button("📡 Scrape", use_container_width=True, help="Pull games + box scores + injuries + Vegas lines. ~2-3 min."):
        with st.spinner("Scraping games, injuries, Vegas..."):
            ok, out = run_pipeline(["nba_playoffs", "fetch_injuries", "fetch_vegas"])
        if ok:
            st.cache_data.clear()
            st.success("Scraped.")
        else:
            st.error("Scrape failed — see log below.")
        with st.expander("Output"):
            st.code(out, language="text")
    if col_b.button("🔮 Predict", use_container_width=True, help="Re-run Monte Carlo bracket simulator. Takes ~5 sec."):
        with st.spinner("Running predict_bracket.py..."):
            ok, out = run_pipeline(["predict_bracket"])
        if ok:
            st.cache_data.clear()
            st.success("Predicted.")
        else:
            st.error("Prediction failed — see log below.")
        with st.expander("Output"):
            st.code(out, language="text")
    if col_c.button("📉 Backtest", use_container_width=True, help="Walk-forward retrain + evaluate on each historical season. ~10 sec."):
        with st.spinner("Running backtest.py..."):
            ok, out = run_pipeline(["backtest"])
        if ok:
            st.cache_data.clear()
            st.success("Backtested.")
        else:
            st.error("Backtest failed — see log below.")
        with st.expander("Output"):
            st.code(out, language="text")

    st.divider()
    data = load_data()
    model = data["model"]
    if model:
        st.markdown("**Model**")
        st.markdown(f"`{model['model_name']}`")
        st.caption(f"Val log_loss: {model['val_metrics']['log_loss']:.4f}")
        st.caption(f"Val accuracy: {model['val_metrics']['accuracy']:.1%}")
        st.caption(f"Test log_loss: {model['test_metrics']['log_loss']:.4f}")
    st.divider()
    inj = data.get("injuries", pd.DataFrame())
    with st.expander("🏥 Injury Report (playoff teams)", expanded=False):
        if inj is None or inj.empty:
            st.caption("No injury data yet — click **📡 Scrape** to fetch.")
        else:
            playoff_teams = set()
            if not data["games"].empty:
                playoff_teams = set(data["games"]["home_team"]).union(data["games"]["away_team"])
            inj_filtered = inj[
                inj["team_abbr"].isin(playoff_teams) & (inj["status"] == "Out")
            ] if playoff_teams else inj[inj["status"] == "Out"]
            if inj_filtered.empty:
                st.caption("No reported Out injuries on playoff teams. 🟢")
            else:
                for tm, grp in inj_filtered.groupby("team_abbr"):
                    color = team_color(tm)
                    st.markdown(
                        f"<div style='color:{color}; font-weight:700; "
                        f"margin-top:8px; margin-bottom:2px'>{tm}</div>",
                        unsafe_allow_html=True,
                    )
                    for _, r in grp.iterrows():
                        st.markdown(
                            f"<div style='color:#C8CDD7; font-size:0.78rem; padding-left:8px'>· {r['player_name']}</div>",
                            unsafe_allow_html=True,
                        )

    st.divider()
    if data["games_path"]:
        gp_date = data["games_path"].stem.split("_")[-1]
        st.caption(f"Games CSV: `{gp_date}`")
    st.caption(f"Today: {datetime.date.today().isoformat()}")


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

games = data["games"]
boxscores = data["boxscores"]
champ = data["champ"]
adv = data["adv"]

st.markdown('<div class="hero-title">NBA Playoffs 2025-26 — Title Race</div>',
            unsafe_allow_html=True)
st.markdown(
    f'<div class="hero-sub">'
    f'<span class="kpi-tag">LIVE</span> &nbsp; '
    f'Model-driven Monte Carlo championship forecast · '
    f'{len(games)} games tracked · updated {datetime.date.today().strftime("%b %-d, %Y")}'
    f'</div>',
    unsafe_allow_html=True,
)
st.write("")

# Cold-start banner if essential artifacts are missing.
if champ.empty or games.empty:
    missing = []
    if games.empty:
        missing.append("**games CSV** (run `📡 Scrape`)")
    if champ.empty:
        missing.append("**champion probabilities** (run `🔮 Predict`)")
    st.warning(
        "Cold start — this dashboard has no data yet.\n\n"
        "Missing: " + " · ".join(missing) +
        "\n\nUse the sidebar buttons (top-left **›** to expand) to generate them. "
        "On Streamlit Cloud first-time setup, click **📡 Scrape** then **🔮 Predict**. "
        "Both take a few minutes total.",
        icon="⚠️",
    )

# Top 3 contender cards (or 3 placeholders if no data yet)
cols = st.columns(3)
if not champ.empty and "champion_prob" in champ.columns:
    top3 = champ.sort_values("champion_prob", ascending=False).head(3).reset_index(drop=True)
    for i, row in top3.iterrows():
        abbr = row["team"]
        prob = row["champion_prob"]
        color = team_color(abbr)
        rank_label = {0: "FAVORITE", 1: "RUNNER-UP", 2: "DARK HORSE"}[i]
        with cols[i]:
            st.markdown(
                f"""
                <div class="contender-card" style="--accent: {color};">
                    <div class="contender-rank">{rank_label}</div>
                    <div class="contender-team">{abbr}</div>
                    <div class="contender-name">{team_name(abbr)}</div>
                    <div class="contender-prob-label">Champion</div>
                    <div class="contender-prob">{prob:.0%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
else:
    for i, rank_label in enumerate(["FAVORITE", "RUNNER-UP", "DARK HORSE"]):
        with cols[i]:
            st.markdown(
                f"""
                <div class="contender-card" style="--accent: #555;">
                    <div class="contender-rank">{rank_label}</div>
                    <div class="contender-team">—</div>
                    <div class="contender-name">awaiting forecast</div>
                    <div class="contender-prob-label">Champion</div>
                    <div class="contender-prob">—</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.write("")
st.write("")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_forecast, tab_bracket, tab_games, tab_teams, tab_backtest = st.tabs(
    ["📊 Forecast", "🌳 Bracket", "🎯 Games", "📈 Teams", "📉 Backtesting"]
)


# ===========================================================================
# Tab 1: Forecast
# ===========================================================================
with tab_forecast:
    if champ.empty or adv.empty:
        st.info("No forecast yet — click **🔮 Predict** in the sidebar to generate champion probabilities. Requires running Scrape first if games CSV is missing.")
    left, right = st.columns([1.1, 1])

    with left:
        st.markdown('<div class="section-title">Championship Probability</div>',
                    unsafe_allow_html=True)
        if champ.empty or "champion_prob" not in champ.columns:
            st.caption("—")
            alive = pd.DataFrame(columns=["team", "champion_prob"])
        else:
            alive = champ[champ["champion_prob"] > 0].sort_values("champion_prob")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=alive["champion_prob"] * 100,
            y=alive["team"],
            orientation="h",
            marker=dict(
                color=[team_color(t) for t in alive["team"]],
                line=dict(color="rgba(255,255,255,0.12)", width=1),
            ),
            text=[f"{p:.1%}" for p in alive["champion_prob"]],
            textposition="outside",
            textfont=dict(color="white", size=13, family="Inter,sans-serif"),
            hovertemplate="<b>%{y}</b><br>P(Champion) = %{x:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            height=max(280, 40 * len(alive) + 60),
            margin=dict(l=10, r=60, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                       zeroline=False, color="#8B95A7",
                       title=dict(text="P(Champion) %", font=dict(color="#8B95A7"))),
            yaxis=dict(showgrid=False, color="white",
                       tickfont=dict(size=13, family="Inter,sans-serif")),
            showlegend=False,
        )
        st.plotly_chart(fig)

    with right:
        st.markdown('<div class="section-title">Round Advancement</div>',
                    unsafe_allow_html=True)
        if adv.empty or "p_made_R2" not in adv.columns:
            st.caption("—")
        else:
            adv_disp = adv[adv["p_made_R2"] > 0].copy().sort_values("p_champion", ascending=False)
            adv_disp = adv_disp[["team", "p_made_R2", "p_made_CF", "p_made_Finals", "p_champion"]]
            adv_disp.columns = ["Team", "Made R2", "Conf Finals", "Finals", "Champion"]
            styled = adv_disp.style.format({
                "Made R2": "{:.0%}", "Conf Finals": "{:.0%}",
                "Finals": "{:.0%}", "Champion": "{:.0%}",
            }).background_gradient(
                subset=["Made R2", "Conf Finals", "Finals", "Champion"],
                cmap="Oranges", vmin=0, vmax=1,
            ).set_properties(**{"font-weight": "600", "color": "#111"})
            st.dataframe(styled, use_container_width=True, hide_index=True, height=380)

    st.markdown('<div class="section-title">Model Notes</div>',
                unsafe_allow_html=True)
    st.markdown(
        "- 10,000 Monte Carlo simulations of each remaining series, game by game.\n"
        "- Each game uses a logistic-regression model trained on **1,172 rows / 7 seasons** of playoff features "
        "(team advanced stats, rolling form, series context, rest, star availability).\n"
        "- Validation log loss: **0.654** (baseline 0.693). Calibration tracks the diagonal "
        "with slight overconfidence in 0.7+ predictions."
    )


# ===========================================================================
# Tab 2: Bracket
# ===========================================================================
with tab_bracket:
    if games.empty:
        st.info("No bracket data yet — click **📡 Scrape** in the sidebar to pull the latest playoff games.")
    st.markdown('<div class="section-title">Live Bracket — Win Probability per Series</div>',
                unsafe_allow_html=True)

    # Aggregate live series state from games.
    def series_state(rnd: int, slot: int) -> dict | None:
        sub = games[(games["round"] == rnd) & (games["slot"] == slot)]
        if sub.empty:
            return None
        finals = sub[sub["status"] == "Final"]
        wins: dict[str, int] = {}
        for _, r in finals.iterrows():
            w = r["home_team"] if r["home_score"] > r["away_score"] else r["away_team"]
            wins[w] = wins.get(w, 0) + 1
        teams_pair = tuple(sorted([sub.iloc[0]["home_team"], sub.iloc[0]["away_team"]]))
        for t in teams_pair:
            wins.setdefault(t, 0)
        winner = None
        for t, w in wins.items():
            if w >= 4:
                winner = t
                break
        return {"teams": teams_pair, "wins": wins, "winner": winner,
                "scheduled": len(sub[sub["status"] == "Scheduled"])}

    def advancement_for(team: str, target_round: int) -> float:
        """Return P(team reaches target_round) from adv df. target_round: 2=R2, 3=CF, 4=Finals, 5=Champion."""
        if adv.empty or team not in adv["team"].values:
            return 0.0
        row = adv[adv["team"] == team].iloc[0]
        return {2: row["p_made_R2"], 3: row["p_made_CF"],
                4: row["p_made_Finals"], 5: row["p_champion"]}.get(target_round, 0.0)

    def render_series_card(rnd: int, slot: int, target_round_for_prob: int,
                           target_label: str) -> None:
        s = series_state(rnd, slot)
        if not s:
            st.markdown(
                '<div class="series-card" style="opacity:0.4">'
                '<div class="series-meta">TBD</div>'
                '<div class="series-row"><span class="series-team">—</span></div>'
                '<div class="series-row"><span class="series-team">—</span></div>'
                '</div>',
                unsafe_allow_html=True,
            )
            return
        a, b = s["teams"]
        wa, wb = s["wins"][a], s["wins"][b]
        winner = s["winner"]
        status = "DONE" if winner else f"BO7 · G{wa+wb+1}"
        pa = advancement_for(a, target_round_for_prob) * 100
        pb = advancement_for(b, target_round_for_prob) * 100
        # Sort so the higher win-prob team is shown on top
        rows = sorted([(a, wa, pa), (b, wb, pb)], key=lambda x: -x[2])
        done_class = " done" if winner else ""
        rows_html = ""
        for tm, w, p in rows:
            color = team_color(tm)
            won_dot = "●" if winner == tm else "○"
            won_color = color if winner == tm else "#3A4151"
            rows_html += (
                f'<div class="series-row">'
                f'  <span class="series-team" style="color:{color}">'
                f'    <span style="color:{won_color}">{won_dot}</span> {tm}'
                f'  </span>'
                f'  <span class="series-score">{w} · {p:.0f}%</span>'
                f'</div>'
            )
        st.markdown(
            f'<div class="series-card{done_class}">'
            f'  <div class="series-meta">R{rnd} · Slot {slot} · {status} · {target_label}</div>'
            f'  {rows_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # East/West layout in 4 columns (R1, R2, CF, Finals)
    st.markdown("##### Eastern Conference")
    e_r1 = [0, 1, 2, 3]  # R1 slots for East
    east_cols = st.columns([1, 1, 1, 1])
    with east_cols[0]:
        st.caption("First Round")
        for slot in e_r1:
            render_series_card(1, slot, 2, "→ R2 prob")
    with east_cols[1]:
        st.caption("Conf. Semis")
        for slot in [0, 1]:
            render_series_card(2, slot, 3, "→ CF prob")
            st.markdown('<div style="height:46px"></div>', unsafe_allow_html=True)
    with east_cols[2]:
        st.caption("Conf. Finals")
        st.markdown('<div style="height:48px"></div>', unsafe_allow_html=True)
        render_series_card(3, 0, 4, "→ Finals prob")
    with east_cols[3]:
        st.caption("NBA Finals")
        st.markdown('<div style="height:110px"></div>', unsafe_allow_html=True)
        render_series_card(4, 0, 5, "→ Champion")

    st.markdown("##### Western Conference")
    w_r1 = [4, 5, 6, 7]
    west_cols = st.columns([1, 1, 1, 1])
    with west_cols[0]:
        st.caption("First Round")
        for slot in w_r1:
            render_series_card(1, slot, 2, "→ R2 prob")
    with west_cols[1]:
        st.caption("Conf. Semis")
        for slot in [2, 3]:
            render_series_card(2, slot, 3, "→ CF prob")
            st.markdown('<div style="height:46px"></div>', unsafe_allow_html=True)
    with west_cols[2]:
        st.caption("Conf. Finals")
        st.markdown('<div style="height:48px"></div>', unsafe_allow_html=True)
        render_series_card(3, 1, 4, "→ Finals prob")
    with west_cols[3]:
        st.caption("(Finals shown above)")


# ===========================================================================
# Tab 3: Games
# ===========================================================================
with tab_games:
    if games.empty:
        st.info("No games yet — click **📡 Scrape** in the sidebar.")
    today = pd.Timestamp(datetime.date.today())
    today_games = games[games["game_date_et"].dt.normalize() == today]
    next_games = games[(games["status"] == "Scheduled") &
                       (games["game_date_et"].dt.normalize() >= today)].sort_values("game_date_et")
    recent_games = games[games["status"] == "Final"].sort_values("game_date_et", ascending=False).head(8)

    cgames_l, cgames_r = st.columns([1.1, 1])

    with cgames_l:
        st.markdown('<div class="section-title">Upcoming Games</div>', unsafe_allow_html=True)
        if next_games.empty:
            st.info("No scheduled games.")
        else:
            for _, r in next_games.head(8).iterrows():
                home, away = r["home_team"], r["away_team"]
                tip = r["tipoff_et"]
                series = r["series"] if pd.notna(r.get("series")) else ""
                label = r["game_label"] if pd.notna(r.get("game_label")) else ""
                hc, ac = team_color(home), team_color(away)
                st.markdown(
                    f"""
                    <div class="series-card" style="margin-bottom:6px">
                      <div class="series-meta">{label} · {series} · {tip}</div>
                      <div class="series-row">
                        <span class="series-team" style="color:{ac}">{away}</span>
                        <span class="series-score">@</span>
                        <span class="series-team" style="color:{hc}">{home}</span>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with cgames_r:
        st.markdown('<div class="section-title">Recent Results</div>', unsafe_allow_html=True)
        if recent_games.empty:
            st.info("No completed games.")
        else:
            for _, r in recent_games.iterrows():
                home, away = r["home_team"], r["away_team"]
                hs, as_ = int(r["home_score"]), int(r["away_score"])
                hc, ac = team_color(home), team_color(away)
                home_w = hs > as_
                weight_h = "800" if home_w else "500"
                weight_a = "500" if home_w else "800"
                opacity_h = "1" if home_w else "0.7"
                opacity_a = "0.7" if home_w else "1"
                date_str = r["game_date_et"].strftime("%b %-d") if pd.notna(r["game_date_et"]) else ""
                st.markdown(
                    f"""
                    <div class="series-card" style="margin-bottom:6px">
                      <div class="series-meta">{date_str}</div>
                      <div class="series-row">
                        <span class="series-team" style="color:{ac}; font-weight:{weight_a}; opacity:{opacity_a}">{away}</span>
                        <span class="series-score">{as_}</span>
                      </div>
                      <div class="series-row">
                        <span class="series-team" style="color:{hc}; font-weight:{weight_h}; opacity:{opacity_h}">{home}</span>
                        <span class="series-score">{hs}</span>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.write("")
    st.markdown('<div class="section-title">Model vs. Vegas — Edge Finder</div>',
                unsafe_allow_html=True)
    upcoming = data["upcoming"]
    vegas = data["vegas"]
    if upcoming.empty:
        st.info("No upcoming-game predictions yet. Click **🔮 Predict** to generate.")
    elif vegas.empty:
        st.info(
            "Vegas lines not loaded. To enable: sign up at https://the-odds-api.com "
            "for a free API key, then `export ODDS_API_KEY=…` and click **📡 Scrape**. "
            "Without odds, the model's win probabilities are shown below."
        )
        # Still show model probs alone
        for _, r in upcoming.iterrows():
            home, away = r["home_team"], r["away_team"]
            ph = r["model_home_win_prob"]; pa = 1 - ph
            hc, ac = team_color(home), team_color(away)
            st.markdown(
                f"""
                <div class="series-card" style="margin-bottom:6px">
                  <div class="series-meta">{r['game_label']} · {r['series']} · {r['tipoff_et']}</div>
                  <div class="series-row">
                    <span class="series-team" style="color:{hc}">{home}</span>
                    <span class="series-score">{ph:.0%} model</span>
                  </div>
                  <div class="series-row">
                    <span class="series-team" style="color:{ac}">{away}</span>
                    <span class="series-score">{pa:.0%} model</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        # Join: match on (home_team, away_team). Pick the closest in time if duplicates.
        merged = upcoming.merge(
            vegas[["home_team", "away_team", "bookmaker",
                   "home_american_odds", "away_american_odds",
                   "home_fair_prob", "away_fair_prob", "vig"]],
            on=["home_team", "away_team"], how="left",
        )
        matched = merged[merged["bookmaker"].notna()]
        unmatched = merged[merged["bookmaker"].isna()]

        if not matched.empty:
            for _, r in matched.iterrows():
                home, away = r["home_team"], r["away_team"]
                hc, ac = team_color(home), team_color(away)
                p_model_h = r["model_home_win_prob"]
                p_vegas_h = r["home_fair_prob"]
                edge_h = p_model_h - p_vegas_h
                sharp = abs(edge_h) > 0.05
                edge_color = "#4ade80" if edge_h > 0.05 else ("#f87171" if edge_h < -0.05 else "#888")
                edge_label = "MODEL LIKES " + home if edge_h > 0.05 else (
                    "MODEL LIKES " + away if edge_h < -0.05 else "MARKET ALIGNED"
                )
                ho = int(r["home_american_odds"])
                ao = int(r["away_american_odds"])
                ho_str = f"+{ho}" if ho > 0 else f"{ho}"
                ao_str = f"+{ao}" if ao > 0 else f"{ao}"
                st.markdown(
                    f"""
                    <div class="series-card" style="margin-bottom:6px; {'border-left: 3px solid ' + edge_color + ';' if sharp else ''}">
                      <div class="series-meta">
                        {r['game_label']} · {r['series']} · {r['tipoff_et']} ·
                        <span style="color:{edge_color}; font-weight:700">{edge_label}</span>
                        <span style="color:#6B7280"> · {r['bookmaker']}</span>
                      </div>
                      <div style="display:grid; grid-template-columns: 1.4fr 1fr 1fr 1fr; gap:8px; align-items:center; padding-top:4px">
                        <div><span class="series-team" style="color:{hc}">{home}</span></div>
                        <div style="text-align:right; color:#C8CDD7"><b>{p_model_h:.0%}</b> model</div>
                        <div style="text-align:right; color:#C8CDD7"><b>{p_vegas_h:.0%}</b> vegas</div>
                        <div style="text-align:right; color:#8B95A7">{ho_str}</div>
                      </div>
                      <div style="display:grid; grid-template-columns: 1.4fr 1fr 1fr 1fr; gap:8px; align-items:center">
                        <div><span class="series-team" style="color:{ac}">{away}</span></div>
                        <div style="text-align:right; color:#C8CDD7"><b>{1-p_model_h:.0%}</b> model</div>
                        <div style="text-align:right; color:#C8CDD7"><b>{1-p_vegas_h:.0%}</b> vegas</div>
                        <div style="text-align:right; color:#8B95A7">{ao_str}</div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if not unmatched.empty:
            with st.expander(f"⚠️  {len(unmatched)} upcoming game(s) without matching Vegas line"):
                for _, r in unmatched.iterrows():
                    st.caption(f"{r['away_team']} @ {r['home_team']} — {r['tipoff_et']}")

    st.write("")
    st.markdown('<div class="section-title">Box Score Explorer</div>', unsafe_allow_html=True)
    if boxscores.empty:
        st.info("No box scores available.")
    else:
        # Build game labels for selection
        completed = games[games["status"] == "Final"].sort_values("game_date_et", ascending=False)
        options = {}
        for _, r in completed.iterrows():
            label = (f"{r['game_date_et'].strftime('%b %-d')} — "
                     f"{r['away_team']} {int(r['away_score'])} @ "
                     f"{r['home_team']} {int(r['home_score'])}")
            options[label] = r["game_id_str"]
        if options:
            picked = st.selectbox("Pick a game", list(options.keys()))
            gid = options[picked]
            box = boxscores[boxscores["game_id_str"] == gid].copy()
            box = box.sort_values(["team_abbr", "pts"], ascending=[True, False])
            display_cols = ["team_abbr", "player_name", "minutes", "pts", "reb", "ast",
                            "stl", "blk", "tov", "pf", "fg_pct", "three_pct", "plus_minus"]
            box_disp = box[[c for c in display_cols if c in box.columns]].rename(columns={
                "team_abbr": "Team", "player_name": "Player", "minutes": "MIN",
                "pts": "PTS", "reb": "REB", "ast": "AST", "stl": "STL", "blk": "BLK",
                "tov": "TOV", "pf": "PF", "fg_pct": "FG%", "three_pct": "3P%",
                "plus_minus": "+/-",
            })
            styled = box_disp.style.format({
                "PTS": "{:.0f}", "REB": "{:.0f}", "AST": "{:.0f}", "STL": "{:.0f}",
                "BLK": "{:.0f}", "TOV": "{:.0f}", "PF": "{:.0f}",
                "FG%": "{:.0%}", "3P%": "{:.0%}", "+/-": "{:+.0f}",
            }, na_rep="—").background_gradient(
                subset=["PTS"], cmap="Oranges", vmin=0, vmax=40,
            ).set_properties(**{"color": "#111"})
            st.dataframe(styled, use_container_width=True, hide_index=True, height=520)


# ===========================================================================
# Tab 4: Teams
# ===========================================================================
with tab_teams:
    if games.empty:
        st.info("No team data yet — click **📡 Scrape** in the sidebar.")
    # Pull current-season team advanced & regular-season leaders from games CSV.
    cleft, cright = st.columns([1.2, 1])

    with cleft:
        st.markdown('<div class="section-title">Regular-Season Scoring Leader by Team (PPG)</div>',
                    unsafe_allow_html=True)
        # Use the games CSV's home/away_reg_season_leader columns (every team appears at least once).
        rs_rows = []
        seen: set[str] = set()
        for _, r in games.iterrows():
            for side in ("home", "away"):
                team = r[f"{side}_team"]
                leader = r.get(f"{side}_reg_season_leader")
                ppg = r.get(f"{side}_reg_season_ppg")
                if team in seen or pd.isna(leader):
                    continue
                seen.add(team)
                rs_rows.append({"team": team, "leader": leader, "ppg": float(ppg)})
        rs_df = pd.DataFrame(rs_rows)
        if not rs_df.empty:
            rs_df = rs_df.sort_values("ppg", ascending=False)
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=rs_df["ppg"], y=rs_df["team"],
                orientation="h",
                marker=dict(color=[team_color(t) for t in rs_df["team"]]),
                text=[f"{l} · {p:.1f}" for l, p in zip(rs_df["leader"], rs_df["ppg"])],
                textposition="outside",
                textfont=dict(color="white", size=11),
                hovertemplate="<b>%{y}</b>: %{text}<extra></extra>",
            ))
            fig2.update_layout(
                height=max(420, 26 * len(rs_df)),
                margin=dict(l=10, r=180, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                           zeroline=False, color="#8B95A7",
                           title=dict(text="PPG", font=dict(color="#8B95A7"))),
                yaxis=dict(showgrid=False, color="white", autorange="reversed"),
                showlegend=False,
            )
            fig2.update_xaxes(range=[0, max(rs_df["ppg"]) * 1.15])
            st.plotly_chart(fig2)

    with cright:
        st.markdown('<div class="section-title">Series Tracker (Active)</div>',
                    unsafe_allow_html=True)
        # Show all in-progress / scheduled series.
        active_keys = []
        for (rnd, slot), grp in games.groupby(["round", "slot"]):
            sub = grp
            finals = sub[sub["status"] == "Final"]
            wins: dict[str, int] = {}
            for _, r in finals.iterrows():
                w = r["home_team"] if r["home_score"] > r["away_score"] else r["away_team"]
                wins[w] = wins.get(w, 0) + 1
            teams_pair = tuple(sorted([sub.iloc[0]["home_team"], sub.iloc[0]["away_team"]]))
            done = any(v >= 4 for v in wins.values())
            if not done:
                active_keys.append((rnd, slot, teams_pair, wins))

        if not active_keys:
            st.info("No active series — bracket is settled or hasn't started.")
        else:
            for rnd, slot, (a, b), wins in active_keys:
                wa = wins.get(a, 0); wb = wins.get(b, 0)
                pa = adv[adv["team"] == a]["p_made_CF"].iloc[0] if a in adv["team"].values else 0
                pb = adv[adv["team"] == b]["p_made_CF"].iloc[0] if b in adv["team"].values else 0
                # Pick advance target based on round
                target = {1: "R2", 2: "CF", 3: "Finals", 4: "Champion"}[rnd]
                ca, cb = team_color(a), team_color(b)
                st.markdown(
                    f"""
                    <div class="series-card">
                      <div class="series-meta">R{rnd} · {target} · G{wa+wb+1}</div>
                      <div class="series-row">
                        <span class="series-team" style="color:{ca}">{a}</span>
                        <span class="series-score">{wa}</span>
                      </div>
                      <div class="series-row">
                        <span class="series-team" style="color:{cb}">{b}</span>
                        <span class="series-score">{wb}</span>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ===========================================================================
# Tab 5: Backtesting — walk-forward honest evaluation
# ===========================================================================
with tab_backtest:
    bt_results = data["bt_results"]
    bt_round = data["bt_round"]
    bt_calib = data["bt_calib"]

    if bt_results.empty:
        st.info(
            "No backtest results yet. Click **📉 Backtest** in the sidebar to run "
            "walk-forward evaluation on every historical season."
        )
    else:
        # Pivot to one row per (season), columns per model
        results_wide = bt_results.pivot_table(
            index=["season", "n_games"],
            columns="model",
            values=["log_loss", "brier", "accuracy"],
        ).reset_index()

        # KPI summary across all backtested seasons
        avg = bt_results.groupby("model")[["log_loss", "brier", "accuracy"]].mean()
        st.markdown('<div class="section-title">Walk-Forward Evaluation Summary</div>',
                    unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Avg Model Log Loss",
                      f"{avg.loc['model', 'log_loss']:.4f}",
                      delta=f"{avg.loc['model', 'log_loss'] - avg.loc['baseline_p50', 'log_loss']:+.4f} vs P=0.5",
                      delta_color="inverse")
        with k2:
            st.metric("Avg Model Accuracy",
                      f"{avg.loc['model', 'accuracy']:.1%}",
                      delta=f"{(avg.loc['model', 'accuracy'] - 0.5) * 100:+.1f} pp vs chance")
        with k3:
            st.metric("Avg Higher-NR Baseline",
                      f"{avg.loc['baseline_higher_nr', 'log_loss']:.4f}",
                      help="\"Always pick the team with higher reg-season net rating at P=0.62\"")
        with k4:
            seasons_beat = (
                bt_results[bt_results["model"] == "model"]
                .merge(bt_results[bt_results["model"] == "baseline_p50"][["season", "log_loss"]],
                       on="season", suffixes=("", "_chance"))
            )
            n_beat = (seasons_beat["log_loss"] < seasons_beat["log_loss_chance"]).sum()
            n_total = len(seasons_beat)
            st.metric("Seasons Beat Chance", f"{n_beat} / {n_total}",
                      help="Number of backtested seasons where model log loss < 0.6931")

        st.write("")

        # Per-season comparison table
        st.markdown('<div class="section-title">Per-Season Comparison</div>',
                    unsafe_allow_html=True)
        table_left, table_right = st.columns([1.0, 1.2])
        with table_left:
            display = bt_results.copy()
            display["model_label"] = display["model"].map({
                "model": "Bracketology",
                "baseline_p50": "Chance (P=0.5)",
                "baseline_higher_nr": "Higher Net Rating",
            })
            t = display.pivot_table(
                index="season", columns="model_label", values="log_loss"
            )[["Bracketology", "Higher Net Rating", "Chance (P=0.5)"]]
            t.columns = [f"{c} ll" for c in t.columns]
            t_styled = t.style.format("{:.4f}").background_gradient(
                cmap="RdYlGn_r", vmin=0.5, vmax=0.9, axis=None
            ).set_properties(**{"font-weight": "600", "color": "#111"})
            st.dataframe(t_styled, use_container_width=True, height=300)
            st.caption("Lower is better. Red = worse than chance.")

        with table_right:
            # Log loss by season — line chart, 3 series
            model_map = {
                "model": ("Bracketology", "#FB8332"),
                "baseline_higher_nr": ("Higher Net Rating", "#8AB4F8"),
                "baseline_p50": ("Chance (P=0.5)", "#666666"),
            }
            fig = go.Figure()
            for key, (label, color) in model_map.items():
                sub = bt_results[bt_results["model"] == key].sort_values("season")
                fig.add_trace(go.Scatter(
                    x=sub["season"], y=sub["log_loss"],
                    mode="lines+markers", name=label,
                    line=dict(color=color, width=3),
                    marker=dict(size=9, color=color, line=dict(color="white", width=1)),
                    hovertemplate=f"<b>{label}</b><br>%{{x}}: ll=%{{y:.4f}}<extra></extra>",
                ))
            fig.update_layout(
                height=320, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, color="#8B95A7", title=""),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                           color="#8B95A7", title=dict(text="Log Loss", font=dict(color="#8B95A7"))),
                legend=dict(orientation="h", yanchor="top", y=1.12, xanchor="right", x=1,
                            font=dict(color="white", size=11),
                            bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig)

        st.write("")

        # Per-round Brier + calibration
        round_left, round_right = st.columns([1, 1])
        with round_left:
            st.markdown('<div class="section-title">Brier Score by Playoff Round</div>',
                        unsafe_allow_html=True)
            if not bt_round.empty:
                # Average Brier per round across all backtested seasons
                round_avg = (
                    bt_round.groupby("round_label", sort=False)["model_brier"]
                    .agg(["mean", "std", "count"])
                    .reset_index()
                )
                round_avg["round_order"] = round_avg["round_label"].map(
                    {"R1": 1, "Conf Semis": 2, "Conf Finals": 3, "NBA Finals": 4}
                )
                round_avg = round_avg.sort_values("round_order")
                fig_r = go.Figure()
                fig_r.add_trace(go.Bar(
                    x=round_avg["round_label"], y=round_avg["mean"],
                    marker=dict(color="#FB8332",
                                line=dict(color="rgba(255,255,255,0.12)", width=1)),
                    error_y=dict(type="data", array=round_avg["std"].fillna(0),
                                 color="rgba(255,255,255,0.3)"),
                    text=[f"{m:.3f}" for m in round_avg["mean"]],
                    textposition="outside", textfont=dict(color="white"),
                    hovertemplate="<b>%{x}</b><br>Brier=%{y:.4f}<extra></extra>",
                ))
                fig_r.add_hline(y=0.25, line_dash="dash", line_color="rgba(255,255,255,0.2)",
                                annotation_text="P=0.5 baseline (0.25)",
                                annotation_font_color="rgba(255,255,255,0.5)")
                fig_r.update_layout(
                    height=320, margin=dict(l=10, r=10, t=20, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False, color="#8B95A7"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#8B95A7",
                               title=dict(text="Brier Score (lower = better)",
                                          font=dict(color="#8B95A7"))),
                )
                st.plotly_chart(fig_r)
                st.caption("Error bars show ±1 SD across seasons. Smaller bars = better model performance.")

        with round_right:
            st.markdown('<div class="section-title">Calibration Curve</div>',
                        unsafe_allow_html=True)
            if not bt_calib.empty:
                fig_c = go.Figure()
                fig_c.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines",
                    line=dict(color="rgba(255,255,255,0.25)", dash="dash"),
                    name="Perfect", hoverinfo="skip",
                ))
                fig_c.add_trace(go.Scatter(
                    x=bt_calib["predicted_avg"], y=bt_calib["observed_avg"],
                    mode="lines+markers",
                    line=dict(color="#FB8332", width=3),
                    marker=dict(size=10 + np.log1p(bt_calib["n"]) * 2,
                                color="#FB8332",
                                line=dict(color="white", width=1)),
                    text=[f"n={int(n)}" for n in bt_calib["n"]],
                    hovertemplate="Predicted %{x:.2f}<br>Observed %{y:.2f}<br>%{text}<extra></extra>",
                    name="Bracketology",
                ))
                fig_c.update_layout(
                    height=320, margin=dict(l=10, r=10, t=20, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#8B95A7", range=[0, 1],
                               title=dict(text="Predicted Probability",
                                          font=dict(color="#8B95A7"))),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#8B95A7", range=[0, 1],
                               title=dict(text="Observed Win Rate",
                                          font=dict(color="#8B95A7"))),
                    showlegend=False,
                )
                st.plotly_chart(fig_c)
                st.caption("Marker size = number of games in bin. Closer to dashed diagonal = better calibrated.")

        st.write("")
        st.markdown('<div class="section-title">What This Means</div>',
                    unsafe_allow_html=True)
        avg_model = avg.loc["model", "log_loss"]
        avg_chance = avg.loc["baseline_p50", "log_loss"]
        avg_nr = avg.loc["baseline_higher_nr", "log_loss"]
        verdict_color = "#4ade80" if avg_model < avg_chance else "#fbbf24"
        verdict_text = (
            "beats" if avg_model < avg_chance
            else "underperforms"
        )
        nr_text = (
            "beats" if avg_model < avg_nr
            else "underperforms"
        )
        st.markdown(
            f"""
            - **Methodology:** for each season, retrain a logistic regression on **only prior seasons** (no look-ahead), then predict every game in that target season.
            - **Avg model log loss: <span style="color:{verdict_color}">{avg_model:.4f}</span>** — model **{verdict_text}** the P=0.5 chance baseline ({avg_chance:.4f}).
            - **Higher-net-rating baseline:** {avg_nr:.4f} — the model **{nr_text}** this simple heuristic on average.
            - Round-1 games are the hardest (most uncertain matchups, biggest Brier). Conf Semis & Conf Finals are typically better-predicted.
            - **Honest read:** the model adds signal in some seasons (2020-21, 2023-24) and overcomplicates in others. With only ~1,200 training rows the marginal features beyond net-rating differential are noisy. Future improvements (live injuries, player-level features, more seasons) should narrow this gap.
            """,
            unsafe_allow_html=True,
        )

st.write("")
st.caption(
    "Data: stats.nba.com via nba_api. Model trained on 7 seasons of playoff data "
    "(2017-18 → 2024-25, bubble skipped). Forecast is for informational/entertainment purposes only."
)
