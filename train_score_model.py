"""Train a score-prediction model for upcoming playoff games.

Predicts (pts_team, pts_opp) jointly from the same features the win-probability
model uses. Compared two model families and picks by validation MAE:
    1. Multi-output linear regression  (interpretable baseline)
    2. Multi-output XGBoost regressor  (typically wins on small tabular data)

Season-based split matches `train_model.py`:
    train:  2017-18, 2018-19, 2020-21, 2021-22, 2022-23
    val:    2023-24
    test:   2024-25

Saves the best model to score_model.pkl (joblib) along with feature column order.

Run:
    python3 train_score_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FEATURES_CSV = DATA / "historical_playoff_features.csv"
MODEL_PATH = ROOT / "score_model.pkl"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

TARGET_COLS = ["pts_team", "pts_opp"]
# Same set of identifier / non-numeric columns as train_model.py, plus targets.
DROP_COLS = {"season", "game_id", "game_date", "team_id", "opp_team_id", "won",
             "pts_team", "pts_opp"}

TRAIN_SEASONS = {"2017-18", "2018-19", "2020-21", "2021-22", "2022-23"}
VAL_SEASONS = {"2023-24"}
TEST_SEASONS = {"2024-25"}


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def load_data() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(FEATURES_CSV)
    for c in ("days_rest_team", "days_rest_opp", "rest_diff"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, feature_cols


def split(df: pd.DataFrame, feature_cols: list[str]):
    def _xy(sub):
        X = sub[feature_cols].to_numpy(dtype=float)
        y = sub[TARGET_COLS].to_numpy(dtype=float)
        return X, y
    return (
        _xy(df[df["season"].isin(TRAIN_SEASONS)]),
        _xy(df[df["season"].isin(VAL_SEASONS)]),
        _xy(df[df["season"].isin(TEST_SEASONS)]),
    )


def metric_block(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-target + combined MAE/RMSE. y is (n, 2): col 0 = pts_team, col 1 = pts_opp."""
    out = {}
    for i, col in enumerate(TARGET_COLS):
        out[f"mae_{col}"] = float(mean_absolute_error(y_true[:, i], y_pred[:, i]))
        out[f"rmse_{col}"] = float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
    out["mae_avg"] = (out["mae_pts_team"] + out["mae_pts_opp"]) / 2
    out["mae_margin"] = float(mean_absolute_error(
        y_true[:, 0] - y_true[:, 1],
        y_pred[:, 0] - y_pred[:, 1],
    ))
    log(f"  {name:>8s} | MAE_team={out['mae_pts_team']:.2f}  "
        f"MAE_opp={out['mae_pts_opp']:.2f}  MAE_avg={out['mae_avg']:.2f}  "
        f"MAE_margin={out['mae_margin']:.2f}")
    return out


def fit_linear(X_tr, y_tr) -> Pipeline:
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("reg", LinearRegression()),
    ])
    pipe.fit(X_tr, y_tr)
    return pipe


def fit_xgb(X_tr, y_tr, X_va, y_va) -> MultiOutputRegressor:
    """XGB has native multi-output but the cleaner path is MultiOutputRegressor
    wrapping single-target XGBRegressors (easier to early-stop per target).
    """
    # Build the wrapper, then call .fit; sklearn's MultiOutputRegressor doesn't
    # forward eval_set to the underlying estimator, so we fit each target
    # ourselves with early stopping for clarity.
    estimators = []
    for i, target_name in enumerate(TARGET_COLS):
        m = xgb.XGBRegressor(
            n_estimators=600,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="reg:squarederror",
            eval_metric="mae",
            early_stopping_rounds=30,
            random_state=42,
            tree_method="hist",
        )
        m.fit(X_tr, y_tr[:, i], eval_set=[(X_va, y_va[:, i])], verbose=False)
        log(f"  xgb {target_name}: best iteration = {m.best_iteration}")
        estimators.append(m)
    # Wrap in a thin object that mimics MultiOutputRegressor's .predict
    class _MultiXGB:
        def __init__(self, models): self.models = models
        def predict(self, X):
            return np.column_stack([m.predict(X) for m in self.models])
    return _MultiXGB(estimators)


def main() -> int:
    if not FEATURES_CSV.exists():
        log(f"Missing {FEATURES_CSV}. Run `python3 build_features.py` first.")
        return 1

    df, feature_cols = load_data()
    if not all(c in df.columns for c in TARGET_COLS):
        log(f"Targets {TARGET_COLS} missing from features CSV. Re-run "
            f"build_features.py to regenerate with pts_team / pts_opp columns.")
        return 1
    log(f"Loaded {len(df)} rows; {len(feature_cols)} features; "
        f"targets={TARGET_COLS}")

    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split(df, feature_cols)
    log(f"Train: {len(X_tr)}  Val: {len(X_va)}  Test: {len(X_te)}")
    naive_mean = y_tr.mean(axis=0)
    log(f"Naive baseline (predict train mean: {naive_mean[0]:.1f} / {naive_mean[1]:.1f}):")
    baseline_pred = np.tile(naive_mean, (len(y_va), 1))
    base_val = metric_block("mean_va", y_va, baseline_pred)

    log("\n--- Linear regression ---")
    lin = fit_linear(X_tr, y_tr)
    lin_val = metric_block("lin_val", y_va, lin.predict(X_va))
    lin_test = metric_block("lin_test", y_te, lin.predict(X_te))

    log("\n--- XGBoost regressor ---")
    xg = fit_xgb(X_tr, y_tr, X_va, y_va)
    xg_val = metric_block("xgb_val", y_va, xg.predict(X_va))
    xg_test = metric_block("xgb_test", y_te, xg.predict(X_te))

    best_name = "linear" if lin_val["mae_avg"] <= xg_val["mae_avg"] else "xgboost"
    best_model = lin if best_name == "linear" else xg
    log(f"\nBest by val MAE_avg: {best_name}")
    log(f"  vs naive baseline: {base_val['mae_avg']:.2f} → "
        f"{(lin_val if best_name == 'linear' else xg_val)['mae_avg']:.2f}")

    joblib.dump({
        "model": best_model,
        "model_name": best_name,
        "feature_cols": feature_cols,
        "target_cols": TARGET_COLS,
        "val_metrics": lin_val if best_name == "linear" else xg_val,
        "test_metrics": lin_test if best_name == "linear" else xg_test,
        "baseline_val_metrics": base_val,
    }, MODEL_PATH)
    log(f"\nSaved {MODEL_PATH}")
    val_m = lin_val if best_name == "linear" else xg_val
    print(f"OK: best={best_name}  val MAE_avg={val_m['mae_avg']:.2f}  "
          f"val MAE_margin={val_m['mae_margin']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
