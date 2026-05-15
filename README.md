# Bracketology — NBA Playoff Forecast

> End-to-end ML pipeline that quantifies championship uncertainty for the NBA playoffs. Scrapes live games, trains a win-probability model on 7 seasons of historical playoff data, and runs 10,000-trial Monte Carlo simulations to project each team's odds of reaching every round.

### 🏀 [**Live dashboard → bracketology-mjg9byoukgakvqnrfcdjtx.streamlit.app**](https://bracketology-mjg9byoukgakvqnrfcdjtx.streamlit.app/)

![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B) ![Model](https://img.shields.io/badge/model-Logistic%20Regression-orange) [![Live](https://img.shields.io/badge/dashboard-live-success)](https://bracketology-mjg9byoukgakvqnrfcdjtx.streamlit.app/)

![Forecast tab — championship probabilities + round advancement matrix](screenshots/01-forecast.png)

---

## Why I built this

The NBA playoffs are widely covered, but most public dashboards either show static seedings (no uncertainty) or pure betting lines (no model). I wanted to build the full loop end-to-end — scraping → feature engineering → modeling → simulation → interactive UI — and see how much signal a modest 1k-row training set can extract from publicly available stats. The result is a working forecast pipeline plus a Streamlit dashboard you can refresh daily.

---

## What it does

- **Scrapes** all 2025-26 playoff games (scheduled, in-progress, completed) plus full traditional box scores from stats.nba.com via [`nba_api`](https://github.com/swar/nba_api).
- **Builds** a 1,172-row historical training set across 7 playoff seasons (2017-18 → 2024-25, bubble skipped) with 49 features per (game, team-perspective) row.
- **Trains** a logistic regression and an XGBoost classifier; selects the best by validation log loss.
- **Simulates** the rest of the bracket 10,000 times, game by game, propagating winners through Conference Semis → Conference Finals → NBA Finals.
- **Visualizes** the live state, the championship probability bar chart, the round-advancement heatmap, the bracket tree, and full box scores in a Streamlit dashboard with NBA team-color branding.

---

## Architecture

```
                        ┌──────────────────────┐
                        │   stats.nba.com      │
                        │   (via nba_api)      │
                        └──────────┬───────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ nba_playoffs.py  │      │ build_features   │      │   cache/         │
│                  │      │ .py              │      │   (parquet pkl   │
│ live games +     │      │                  │      │   responses,     │
│ box scores       │      │ 7 seasons of     │      │   ~600 entries)  │
└────────┬─────────┘      │ playoff features │      └──────────────────┘
         │                └────────┬─────────┘
         │                         │
         │                         ▼
         │               ┌──────────────────┐
         │               │ train_model.py   │
         │               │ LogReg + XGBoost │
         │               │ → model.pkl      │
         │               └────────┬─────────┘
         │                        │
         ▼                        ▼
┌─────────────────────────────────────────┐
│       predict_bracket.py                │
│       10,000 Monte Carlo bracket sims   │
│       → champion_probabilities.csv      │
│       → round_advancement.csv           │
└────────────────┬────────────────────────┘
                 │
                 ▼
        ┌────────────────────┐
        │  dashboard.py      │
        │  (Streamlit)       │
        │  4 tabs · live UI  │
        └────────────────────┘
```

---

## Quickstart

```bash
git clone https://github.com/Bsmit3021/Bracketology.git
cd Bracketology

# Install dependencies (Python 3.9+)
pip install -r requirements.txt

# 1. Scrape current playoff state (~2-3 min)
python3 nba_playoffs.py

# 2. Build historical training set (~20 min first run; instant after, due to cache)
python3 build_features.py

# 3. Train model
python3 train_model.py

# 4. Generate forecast (~5 sec)
python3 predict_bracket.py

# 5. Launch dashboard
python3 -m streamlit run dashboard.py
```

Daily workflow once trained:

```bash
python3 nba_playoffs.py && python3 predict_bracket.py
python3 -m streamlit run dashboard.py
```

The dashboard's sidebar has buttons to re-run the scraper and predictor without leaving the browser.

---

## Dashboard

Four tabs, dark theme, NBA team colors throughout:

| Tab | What it shows |
|---|---|
| 📊 **Forecast** | Horizontal championship-probability bar chart (each team in its primary color), color-graded round-advancement matrix, and the **Finals MVP Watch** leaderboard |
| 🌳 **Bracket** | Eastern + Western conferences laid out across 4 rounds; each series card shows current score, eliminated team (open bullet), and probability to advance |
| 🎯 **Games** | Upcoming games with tip-off times, **predicted scores** (e.g. "CLE 112 - DET 107"), **Vegas line comparison** with edge highlighting, recent results, full box-score explorer |
| 📈 **Teams** | Reg-season scoring leader per team (PPG, team-colored), active-series tracker |
| 📉 **Backtesting** | Walk-forward evaluation with per-season log loss, per-round Brier with error bars, calibration curve, and an honest verdict |

---

## Model Card

### Data
| Split | Seasons | Rows | Games |
|---|---|---|---|
| Train | 2017-18, 2018-19, 2020-21, 2021-22, 2022-23 | 840 | 420 |
| Validation | 2023-24 | 164 | 82 |
| Test (held out) | 2024-25 | 168 | 84 |
| **Total** | 7 seasons | **1,172** | **586** |

2019-20 (Orlando bubble) is excluded — no home-court conditions, not representative of normal playoff dynamics.

### Features (49)
Each row = one (game, team-perspective). Two rows per game (home and away).

- **Series context (6):** wins-so-far, score differential, game number 1–7, elimination flags
- **Rest (3):** days since last playoff game (team & opp), differential
- **Rolling form (10):** last-5-playoff-game avg PTS / PTS allowed / margin / win pct (team & opp)
- **Star availability (2):** count of team's top-3 reg-season scorers playing ≥10 min
- **Team strength (27):** off rating, def rating, net rating, pace, TS%, AST%, OREB%, DREB%, TM_TOV% — each in `team`, `opp`, and `diff` form, from `LeagueDashTeamStats` Advanced/Per100Possessions
- **Home court (1):** is_home flag

### Performance

| Split | Log Loss | Brier | Accuracy |
|---|---|---|---|
| Baseline (P=0.5) | 0.6931 | — | 50.0% |
| **LogReg (val)** ← saved | **0.6538** | 0.229 | 64.6% |
| LogReg (test) | 0.6651 | 0.235 | 59.5% |
| XGBoost (val) | 0.6574 | 0.232 | 65.9% |
| XGBoost (test) | 0.6618 | 0.235 | 60.7% |

Both models beat the 50/50 baseline by ~6%. Test-set log loss is within ~1.5% of validation — modest overfitting but acceptable. Calibration plot tracks the diagonal with slight overconfidence at the 0.7+ end.

### Walk-forward backtest

The dashboard's Backtesting tab retrains the model on prior seasons only and predicts each subsequent season game-by-game. The result is intentionally surfaced honestly, not hidden:

![Backtesting tab — per-round Brier, calibration curve, and honest verdict](screenshots/02-backtesting.png)

- Average walk-forward log loss: **0.712** — *below* the 0.693 chance baseline
- Higher-net-rating heuristic averages **0.672** — outperforms the model on average
- Model beats chance in 3 of 6 backtested seasons; loses in the other 3
- Round-1 games are hardest; Conf Semis & Conf Finals are best-predicted

This is the kind of result most portfolio projects bury. The takeaway: with only ~1,200 training rows, marginal features beyond net-rating differential are noisy. Future improvements (live injury data, player-level features, more seasons) should narrow the gap.

**Top XGBoost feature importances:**
1. `diff_ts_pct` (true-shooting % differential) — 0.075
2. `diff_net_rating` — 0.050
3. `is_home` — 0.038
4. `roll_margin_opp` — 0.032
5. `opp_off_rating` — 0.029

### Limitations
- **No live injury data.** The live simulator hard-codes `stars_active = 3` for every team. Historical training data uses real per-game star presence, but at prediction time we have no injury feed.
- **Static rolling form during simulation.** A team's last-5 rolling stats are computed once from current real games; they don't update as the simulator advances them through hypothetical wins/losses.
- **Modest training set.** ~1,200 rows is small. XGBoost over-fits beyond ~50 trees (early stopping caps it at iteration 45). More seasons or game-level training would help.
- **Static reg-season stats.** Team advanced metrics are season totals; ratings during the playoffs themselves often shift due to defensive intensity / matchup-specific dynamics.
- **Default 2-day rest.** Simulator doesn't track simulated game dates, so rest_diff is fixed at 0.
- **Data source unofficial.** stats.nba.com endpoints used via `nba_api` are not an official public API. Commercial usage has unclear ToS implications.

### Intended Use
This is a **portfolio / educational project**. The forecast is for informational purposes, not betting advice. ~65% game accuracy is competitive with public NBA prediction baselines but well below Vegas closing lines (~67–70% on moneyline favorites).

---

## Repository structure

```
bracketology/
├── nba_playoffs.py          # Scraper — live games + box scores
├── build_features.py        # Historical 7-season training set builder
├── train_model.py           # LogReg + XGBoost, season-based split
├── predict_bracket.py       # Monte Carlo bracket simulator
├── dashboard.py             # Streamlit UI
├── tests/
│   └── test_basics.py       # Unit tests for parsing + invariants
├── requirements.txt
├── LICENSE
├── README.md
└── .gitignore               # excludes cache/, data/*.csv, model.pkl
```

Regenerable artifacts (`cache/`, `data/*.csv`, `model.pkl`, `reports/`) are excluded from version control — anyone cloning the repo runs the pipeline and rebuilds them.

---

## Tech stack

**Data:** `nba_api` (stats.nba.com client) · **Modeling:** scikit-learn (LogisticRegression, StandardScaler, SimpleImputer, calibration_curve), xgboost (gradient boosting w/ early stopping) · **Simulation:** custom Monte Carlo in pure Python · **Viz:** Plotly, Matplotlib · **UI:** Streamlit + custom CSS · **Persistence:** joblib, pickle, pandas

---

## Roadmap

Planned improvements, roughly in impact-per-effort order:

1. **Live injury feed** — pull ESPN injury endpoint, override `stars_active` per simulated game. Biggest single accuracy win available.
2. **Vegas line comparison** — fetch closing moneylines; highlight games where the model disagrees with the market.
3. **Backtesting tab** — replay model on 2023-24 and 2024-25 game-by-game; plot rolling calibration and Brier over time.
4. **Player-level features** — top-3 scorers' rolling PPG / TS% instead of binary active flag.
5. **GitHub Actions cron** — daily scrape + predict at 6am ET, commit refreshed CSVs.
6. **Mobile-responsive CSS** — Streamlit defaults don't scale well to phone.
7. **Multi-sport** — same pipeline against NCAA, WNBA, or NHL playoff data.

---

## License

[MIT](LICENSE). Use, fork, or commercialize freely — just keep the copyright notice. Data scraped from stats.nba.com remains property of the NBA.
