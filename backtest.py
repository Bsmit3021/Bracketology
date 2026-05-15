"""Walk-forward backtesting for Bracketology's win-probability model.

For each backtested season, retrains a logistic-regression model on all
prior-season data only (no look-ahead), then predicts every game in the
target season. Computes:

  * per-season log loss, Brier, accuracy
  * per-(season, round) Brier
  * pooled calibration bins across all backtested seasons
  * comparison vs. two baselines: P=0.5, and "higher-net-rating wins at 0.62"

Outputs three CSVs in data/:
  - backtest_results.csv
  - backtest_round_brier.csv
  - backtest_calibration.csv

Run:
    python3 backtest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FEATURES_CSV = DATA / "historical_playoff_features.csv"

# Same drop list as train_model.py.
DROP_COLS = {"season", "game_id", "game_date", "team_id", "opp_team_id", "won"}

# Seasons in chronological order. Each is backtested by training on those before it.
SEASONS = ["2017-18", "2018-19", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
# We need at least 1 prior season → first backtested season is index 1.
BACKTEST_SEASONS = SEASONS[1:]

# Baseline: "higher net rating wins" — pick probability that roughly matches
# the actual home-team win rate during the playoffs so the log loss is fair.
HIGHER_NET_RATING_PROB = 0.62


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def parse_round(game_id) -> int:
    """Extract playoff round (1-4) from a game_id."""
    s = str(game_id).zfill(10)
    try:
        return int(s[7])
    except (ValueError, IndexError):
        return 0


def load_features() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(FEATURES_CSV)
    # Coerce string-encoded nullable numerics.
    for c in ("days_rest_team", "days_rest_opp", "rest_diff"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["round"] = df["game_id"].apply(parse_round)
    return df, feature_cols


def fit_logreg(X: np.ndarray, y: np.ndarray) -> Pipeline:
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])
    pipe.fit(X, y)
    return pipe


def metrics_block(name: str, y: np.ndarray, p: np.ndarray) -> dict:
    p_clipped = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "model": name,
        "log_loss": float(log_loss(y, p_clipped)),
        "brier": float(brier_score_loss(y, p)),
        "accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
    }


def baseline_higher_net_rating(X_row_diff_net: np.ndarray) -> np.ndarray:
    """Predict HIGHER_NET_RATING_PROB if diff_net_rating > 0 else 1 - that."""
    p = np.where(X_row_diff_net > 0, HIGHER_NET_RATING_PROB, 1 - HIGHER_NET_RATING_PROB)
    return p.astype(float)


def calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Equal-width calibration buckets, plus the actual win rate per bucket."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    rows = []
    for i in range(n_bins):
        mask = idx == i
        if not mask.any():
            continue
        rows.append({
            "bin_lo": float(bins[i]),
            "bin_hi": float(bins[i + 1]),
            "predicted_avg": float(p[mask].mean()),
            "observed_avg": float(y[mask].mean()),
            "n": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def main() -> int:
    if not FEATURES_CSV.exists():
        log(f"Missing {FEATURES_CSV}. Run `python3 build_features.py` first.")
        return 1
    df, feature_cols = load_features()
    log(f"Loaded {len(df)} rows across {df['season'].nunique()} seasons "
        f"({len(feature_cols)} features).")

    season_rows: list[dict] = []
    round_rows: list[dict] = []
    all_y: list[int] = []
    all_p_model: list[float] = []

    for target in BACKTEST_SEASONS:
        train_seasons = [s for s in SEASONS if s < target]
        train_mask = df["season"].isin(train_seasons)
        test_mask = df["season"] == target

        X_tr = df.loc[train_mask, feature_cols].to_numpy(dtype=float)
        y_tr = df.loc[train_mask, "won"].to_numpy(dtype=int)
        X_te = df.loc[test_mask, feature_cols].to_numpy(dtype=float)
        y_te = df.loc[test_mask, "won"].to_numpy(dtype=int)
        rounds_te = df.loc[test_mask, "round"].to_numpy(dtype=int)
        diff_nr_te = df.loc[test_mask, "diff_net_rating"].to_numpy(dtype=float)

        if len(X_tr) == 0 or len(X_te) == 0:
            log(f"  skipping {target}: empty split")
            continue

        model = fit_logreg(X_tr, y_tr)
        p_model = model.predict_proba(X_te)[:, 1]
        p_chance = np.full_like(y_te, 0.5, dtype=float)
        p_nr_baseline = baseline_higher_net_rating(diff_nr_te)

        m_model = metrics_block("model", y_te, p_model)
        m_chance = metrics_block("baseline_p50", y_te, p_chance)
        m_nr = metrics_block("baseline_higher_nr", y_te, p_nr_baseline)

        for m in (m_model, m_chance, m_nr):
            season_rows.append({
                "season": target, "n_games": int(len(y_te)),
                "train_seasons": "+".join(train_seasons),
                **m,
            })

        log(f"  {target}: n={len(y_te)}  model ll={m_model['log_loss']:.4f}  "
            f"acc={m_model['accuracy']:.3f}  brier={m_model['brier']:.4f}")

        # Per-round Brier (model only).
        for r in sorted(set(int(x) for x in rounds_te)):
            rm = rounds_te == r
            if not rm.any():
                continue
            round_rows.append({
                "season": target,
                "round": int(r),
                "round_label": {1: "R1", 2: "Conf Semis", 3: "Conf Finals",
                                4: "NBA Finals"}.get(int(r), f"R{r}"),
                "n_games": int(rm.sum()),
                "model_brier": float(brier_score_loss(y_te[rm], p_model[rm])),
                "model_accuracy": float(accuracy_score(
                    y_te[rm], (p_model[rm] >= 0.5).astype(int))),
            })

        all_y.extend(y_te.tolist())
        all_p_model.extend(p_model.tolist())

    season_df = pd.DataFrame(season_rows)
    round_df = pd.DataFrame(round_rows)
    calib_df = calibration_bins(np.array(all_y), np.array(all_p_model))

    (DATA / "backtest_results.csv").write_text(season_df.to_csv(index=False))
    (DATA / "backtest_round_brier.csv").write_text(round_df.to_csv(index=False))
    (DATA / "backtest_calibration.csv").write_text(calib_df.to_csv(index=False))

    log("\nWrote: backtest_results.csv, backtest_round_brier.csv, backtest_calibration.csv")

    # Print summary block.
    summary = (
        season_df[season_df["model"] == "model"]
        .agg({"log_loss": "mean", "brier": "mean", "accuracy": "mean"})
    )
    chance_summary = (
        season_df[season_df["model"] == "baseline_p50"]
        .agg({"log_loss": "mean", "brier": "mean", "accuracy": "mean"})
    )
    nr_summary = (
        season_df[season_df["model"] == "baseline_higher_nr"]
        .agg({"log_loss": "mean", "brier": "mean", "accuracy": "mean"})
    )
    log("\nAverage across backtested seasons:")
    log(f"  model         log_loss={summary['log_loss']:.4f}  "
        f"brier={summary['brier']:.4f}  acc={summary['accuracy']:.3f}")
    log(f"  P=0.5         log_loss={chance_summary['log_loss']:.4f}  "
        f"brier={chance_summary['brier']:.4f}  acc={chance_summary['accuracy']:.3f}")
    log(f"  higher netrtg log_loss={nr_summary['log_loss']:.4f}  "
        f"brier={nr_summary['brier']:.4f}  acc={nr_summary['accuracy']:.3f}")
    print(f"OK: backtested {len(BACKTEST_SEASONS)} seasons "
          f"(avg model log_loss {summary['log_loss']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
