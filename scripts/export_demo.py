"""
Walk-forward backtest of the Over/Under model, exported for the in-browser demo.

  demo/nfl-games.json   every out-of-sample game: date, teams, line, result, model P(Over)
  demo/nfl-summary.json headline numbers (accuracy, accuracy by confidence, default strategies)

How it stays honest:
  * Hyperparameters and the model family are chosen using only games before the first
    test window, so nothing about the test years leaks into the setup.
  * Each test window is predicted by a model trained only on earlier games.
  * Pushes (total exactly on the line) are refunded in betting and left out of accuracy.
  * Odds are a flat -110 on both sides.

Run:  python -m scripts.export_demo          (FAST=1 skips Optuna for a quick local check)
"""

import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing import DataPreprocessor  # noqa: E402
from src.elo_rating import EloCalculator  # noqa: E402
from src.feature_engineering import FeatureEngineer, get_feature_columns_list  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
FAST = os.environ.get("FAST") == "1"
ODDS = -110
BREAK_EVEN = 110 / 210  # 52.38%
N_SPLITS, TEST_SHARE = 5, 0.10


def load():
    pre = DataPreprocessor("Dataset.xlsx")
    df = pre.clean_and_preprocess(pre.load_data())
    df = FeatureEngineer(elo_calculator=EloCalculator()).compute_all_features(df)
    cols = get_feature_columns_list()
    df = df.dropna(subset=cols + ["Over Under Line", "Score Home", "Score Away"])
    df = df.sort_values("Schedule Date", kind="stable").reset_index(drop=True)
    total = df["Score Home"] + df["Score Away"]
    df["result"] = np.where(total > df["Over Under Line"], 1, np.where(total < df["Over Under Line"], 0, -1))
    return df, cols


def make_lr(params=None):
    return make_pipeline(StandardScaler(), LogisticRegression(C=(params or {}).get("C", 1.0), max_iter=2000))


def make_xgb(params):
    from xgboost import XGBClassifier
    return XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1, **params)


def tune(X, y):
    """Choose model family and hyperparameters on pre-test games only (chronological 80/20)."""
    split = int(len(X) * 0.8)
    Xa, ya, Xb, yb = X[:split], y[:split], X[split:], y[split:]

    def score(model):
        model.fit(Xa, ya)
        return log_loss(yb, np.clip(model.predict_proba(Xb)[:, 1], 1e-6, 1 - 1e-6))

    if FAST:
        return "logistic_regression", {"C": 1.0}, {"logistic_regression": score(make_lr())}

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    lr_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    lr_study.optimize(lambda t: score(make_lr({"C": t.suggest_float("C", 1e-4, 1e2, log=True)})), n_trials=25)

    def xgb_obj(t):
        return score(make_xgb({
            "n_estimators": t.suggest_int("n_estimators", 50, 400),
            "max_depth": t.suggest_int("max_depth", 2, 6),
            "learning_rate": t.suggest_float("learning_rate", 5e-3, 0.2, log=True),
            "subsample": t.suggest_float("subsample", 0.6, 1.0),
            "min_child_weight": t.suggest_int("min_child_weight", 1, 20),
        }))

    xgb_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    xgb_study.optimize(xgb_obj, n_trials=40)

    scores = {"logistic_regression": lr_study.best_value, "xgboost": xgb_study.best_value}
    if scores["xgboost"] < scores["logistic_regression"]:
        return "xgboost", xgb_study.best_params, scores
    return "logistic_regression", lr_study.best_params, scores


def simulate(games, edge, side="both", sizing="flat", stake=50.0, bankroll=5000.0, kelly=0.5):
    """Same rules as the browser demo (demos/nfl-demo.js)."""
    b = 100 / abs(ODDS)
    bank, peak, max_dd, bets, wins, pushes, staked = bankroll, bankroll, 0.0, 0, 0, 0, 0.0
    for p, res in games:
        pick = None
        if side in ("both", "over") and p - BREAK_EVEN >= edge:
            pick, q = 1, p
        elif side in ("both", "under") and (1 - p) - BREAK_EVEN >= edge:
            pick, q = 0, 1 - p
        if pick is None or bank <= 0:
            continue
        size = stake if sizing == "flat" else max(0.0, kelly * (q * (b + 1) - 1) / b) * bank
        size = min(size, bank)
        if size <= 0:
            continue
        bets += 1
        staked += size
        if res == -1:
            pushes += 1
        elif res == pick:
            wins += 1
            bank += size * b
        else:
            bank -= size
        peak = max(peak, bank)
        max_dd = max(max_dd, (peak - bank) / peak if peak > 0 else 0)
    decided = bets - pushes
    return {"bets": bets, "win_rate": wins / decided if decided else None, "final": round(bank, 2),
            "roi": (bank - bankroll) / staked if staked else 0.0, "max_drawdown": max_dd}


def main():
    df, cols = load()
    X = df[cols].values.astype(float)
    y = (df["result"] == 1).astype(int).values
    decided = df["result"].values != -1

    tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=int(len(df) * TEST_SHARE))
    splits = list(tscv.split(X))
    first_test = splits[0][1][0]
    pre = np.arange(first_test)
    pre = pre[decided[pre]]
    family, params, tune_scores = tune(X[pre], y[pre])
    logging.info(f"Chosen on pre-test games: {family} {params} (val log loss {tune_scores})")

    prob = np.full(len(df), np.nan)
    folds = []
    for k, (tr, te) in enumerate(splits):
        tr = tr[decided[tr]]
        model = make_xgb(params) if family == "xgboost" else make_lr(params)
        model.fit(X[tr], y[tr])
        prob[te] = model.predict_proba(X[te])[:, 1]
        m = te[decided[te]]
        acc = accuracy_score(y[m], prob[m] >= 0.5)
        folds.append({"fold": k + 1, "start": str(df["Schedule Date"].iloc[te[0]].date()),
                      "end": str(df["Schedule Date"].iloc[te[-1]].date()), "games": int(len(m)), "accuracy": acc})
        logging.info(f"fold {k + 1}: {acc:.4f} on {len(m)} games")

    oof = ~np.isnan(prob)
    dm = oof & decided
    p_d, y_d = prob[dm], y[dm]
    acc = accuracy_score(y_d, p_d >= 0.5)
    conf = np.abs(p_d - 0.5)
    buckets = []
    for lo, hi in [(0, 0.02), (0.02, 0.04), (0.04, 0.06), (0.06, 0.08), (0.08, 1)]:
        sel = (conf >= lo) & (conf < hi)
        if sel.sum():
            buckets.append({"from": lo, "to": hi, "games": int(sel.sum()),
                            "accuracy": accuracy_score(y_d[sel], p_d[sel] >= 0.5)})

    out = df.loc[oof]
    games = [[str(d.date()), h, a, float(line), int(r), round(float(p), 4)]
             for d, h, a, line, r, p in zip(out["Schedule Date"], out["Team Home"], out["Team Away"],
                                           out["Over Under Line"], out["result"], prob[oof])]
    sim_input = [(g[5], g[4]) for g in games]
    strategies = {
        "flat_edge3": simulate(sim_input, 0.03),
        "flat_edge0": simulate(sim_input, 0.0),
        "half_kelly_edge3": simulate(sim_input, 0.03, sizing="kelly"),
    }
    summary = {
        "model": family, "params": params, "tuning_log_loss": tune_scores,
        "games_total": int(len(df)), "games_tested": int(oof.sum()), "pushes_tested": int((oof & ~decided).sum()),
        "first_test_date": games[0][0], "last_test_date": games[-1][0],
        "accuracy": acc, "break_even": BREAK_EVEN, "over_rate": float(y_d.mean()),
        "log_loss": log_loss(y_d, p_d), "log_loss_coin": log_loss(y_d, np.full(len(y_d), 0.5)),
        "brier": brier_score_loss(y_d, p_d), "folds": folds, "by_confidence": buckets,
        "strategies": strategies,
    }
    os.makedirs("demo", exist_ok=True)
    with open("demo/nfl-games.json", "w") as fh:
        json.dump({"cols": ["date", "home", "away", "line", "result", "p_over"], "games": games}, fh,
                  separators=(",", ":"))
    with open("demo/nfl-summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
