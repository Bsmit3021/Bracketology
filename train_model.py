"""
Train win-probability classifier on historical playoff features (Part B2).

Split:
    train      = 2017-18, 2018-19, 2020-21, 2021-22, 2022-23   (5 seasons)
    validation = 2023-24
    test       = 2024-25  (held out)

Two models compared by validation log loss; best is saved to model.pkl.

Run:
    python3 train_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "historical_playoff_features.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
MODEL_PATH = ROOT / "model.pkl"

TRAIN_SEASONS = {"2017-18", "2018-19", "2020-21", "2021-22", "2022-23"}
VAL_SEASONS = {"2023-24"}
TEST_SEASONS = {"2024-25"}

# Columns that are identifiers / label / non-numeric — drop from features.
DROP_COLS = {"season", "game_id", "game_date", "team_id", "opp_team_id", "won"}


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def load_data() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(DATA)
    # Coerce known nullable-numeric columns (stored as "" for None) to float.
    for c in ("days_rest_team", "days_rest_opp", "rest_diff"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    # Force every feature column to numeric (string-leaking guard).
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, feature_cols


def split(df: pd.DataFrame, feature_cols: list[str]):
    def _xy(sub):
        X = sub[feature_cols].to_numpy(dtype=float)
        y = sub["won"].to_numpy(dtype=int)
        return X, y

    train = df[df["season"].isin(TRAIN_SEASONS)]
    val = df[df["season"].isin(VAL_SEASONS)]
    test = df[df["season"].isin(TEST_SEASONS)]
    log(f"Train rows: {len(train)} | Val rows: {len(val)} | Test rows: {len(test)}")
    return _xy(train), _xy(val), _xy(test)


def report(name: str, y_true, p) -> dict[str, float]:
    pred = (p >= 0.5).astype(int)
    metrics = {
        "log_loss": float(log_loss(y_true, np.clip(p, 1e-6, 1 - 1e-6))),
        "brier": float(brier_score_loss(y_true, p)),
        "accuracy": float(accuracy_score(y_true, pred)),
    }
    log(f"  {name:>10s} | log_loss={metrics['log_loss']:.4f} "
        f"| brier={metrics['brier']:.4f} | acc={metrics['accuracy']:.4f}")
    return metrics


def calibration_plot(name: str, y_true, p, path: Path) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
    ax.plot(mean_pred, frac_pos, "o-", label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration — {name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log(f"  -> calibration plot: {path}")


def train_logreg(X_train, y_train) -> Pipeline:
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_xgb(X_train, y_train, X_val, y_val) -> xgb.XGBClassifier:
    clf = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        tree_method="hist",
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    log(f"  xgb best iteration: {clf.best_iteration}")
    return clf


def main() -> int:
    df, feature_cols = load_data()
    log(f"Loaded {len(df)} rows, {len(feature_cols)} features.")

    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split(df, feature_cols)
    baseline = float(log_loss(y_va, np.full_like(y_va, 0.5, dtype=float)))
    log(f"Validation baseline (P=0.5): log_loss={baseline:.4f}")

    log("\n--- Logistic Regression ---")
    lr = train_logreg(X_tr, y_tr)
    lr_p_va = lr.predict_proba(X_va)[:, 1]
    lr_p_te = lr.predict_proba(X_te)[:, 1]
    lr_val = report("LR val", y_va, lr_p_va)
    lr_test = report("LR test", y_te, lr_p_te)
    calibration_plot("LogReg (val)", y_va, lr_p_va, REPORTS / "calibration_lr_val.png")

    log("\n--- XGBoost ---")
    xg = train_xgb(X_tr, y_tr, X_va, y_va)
    xg_p_va = xg.predict_proba(X_va)[:, 1]
    xg_p_te = xg.predict_proba(X_te)[:, 1]
    xg_val = report("XGB val", y_va, xg_p_va)
    xg_test = report("XGB test", y_te, xg_p_te)
    calibration_plot("XGBoost (val)", y_va, xg_p_va, REPORTS / "calibration_xgb_val.png")

    # Pick best by validation log loss.
    best_name = "logreg" if lr_val["log_loss"] <= xg_val["log_loss"] else "xgboost"
    best_model = lr if best_name == "logreg" else xg
    log(f"\nBest model by val log_loss: {best_name}")

    # Top XGB feature importances (informational).
    if best_name == "xgboost" or True:
        try:
            importances = xg.feature_importances_
            ranked = sorted(zip(feature_cols, importances), key=lambda x: -x[1])[:10]
            log("Top 10 XGB feature importances:")
            for f, imp in ranked:
                log(f"  {f:30s} {imp:.4f}")
        except Exception:
            pass

    joblib.dump(
        {
            "model": best_model,
            "model_name": best_name,
            "feature_cols": feature_cols,
            "val_metrics": {"logreg": lr_val, "xgboost": xg_val}[best_name],
            "test_metrics": {"logreg": lr_test, "xgboost": xg_test}[best_name],
        },
        MODEL_PATH,
    )
    log(f"\nSaved best model to {MODEL_PATH}")
    print(f"OK: best={best_name} val_ll={({'logreg': lr_val, 'xgboost': xg_val}[best_name])['log_loss']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
