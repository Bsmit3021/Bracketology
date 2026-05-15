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

    boxscores = pd.read_csv(box_path) if box_path else pd.DataFrame()
    if not boxscores.empty:
        boxscores["game_id_str"] = boxscores["game_id"].astype(str).str.zfill(10)

    champ = pd.read_csv(champ_path) if champ_path.exists() else pd.DataFrame()
    adv = pd.read_csv(adv_path) if adv_path.exists() else pd.DataFrame()

    bundle = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

    return {
        "games": games,
        "boxscores": boxscores,
        "champ": champ,
        "adv": adv,
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
    col_a, col_b = st.columns(2)
    if col_a.button("📡 Scrape", use_container_width=True, help="Re-pull games + box scores from NBA stats API. Takes ~2-3 min."):
        with st.spinner("Running nba_playoffs.py..."):
            ok, out = run_pipeline(["nba_playoffs"])
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

# Top 3 contender cards
top3 = champ.sort_values("champion_prob", ascending=False).head(3).reset_index(drop=True)
cols = st.columns(3)
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

st.write("")
st.write("")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_forecast, tab_bracket, tab_games, tab_teams = st.tabs(
    ["📊 Forecast", "🌳 Bracket", "🎯 Games", "📈 Teams"]
)


# ===========================================================================
# Tab 1: Forecast
# ===========================================================================
with tab_forecast:
    left, right = st.columns([1.1, 1])

    with left:
        st.markdown('<div class="section-title">Championship Probability</div>',
                    unsafe_allow_html=True)
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
        adv_disp = adv[adv["p_made_R2"] > 0].copy().sort_values("p_champion", ascending=False)
        adv_disp = adv_disp[["team", "p_made_R2", "p_made_CF", "p_made_Finals", "p_champion"]]
        adv_disp.columns = ["Team", "Made R2", "Conf Finals", "Finals", "Champion"]

        # Custom color-graded display
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
        rs_df = pd.DataFrame(rs_rows).sort_values("ppg", ascending=False)
        if not rs_df.empty:
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

st.write("")
st.caption(
    "Data: stats.nba.com via nba_api. Model trained on 7 seasons of playoff data "
    "(2017-18 → 2024-25, bubble skipped). Forecast is for informational/entertainment purposes only."
)
