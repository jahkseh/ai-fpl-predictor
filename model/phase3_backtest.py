#!/usr/bin/env python3
"""Phase 3 controlled walk-forward backtest.

Compares the existing rolling-form model with the new historical-intelligence
feature set on the same unseen 2025/26 gameweeks. No future gameweek data is
used when a prediction is produced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from historical_intelligence import load_completed_seasons
from phase3_features import HISTORICAL_FEATURES, prepare_historical_training
from phase4_features import (
    TEAM_FEATURES,
    build_historical_team_features,
    attach_team_features,
    add_fixture_environment,
)

BASELINE = [
    "previous_3_total_points", "previous_5_total_points",
    "previous_10_total_points", "previous_3_minutes",
    "previous_5_minutes", "previous_10_minutes",
    "previous_5_start_rate", "previous_10_start_rate",
    "previous_5_minutes_rate", "previous_10_minutes_rate",
    "previous_3_expected_goals", "previous_5_expected_goals",
    "previous_10_expected_goals", "previous_3_expected_assists",
    "previous_5_expected_assists", "previous_10_expected_assists",
    "value", "was_home",
]
PHASE3 = BASELINE + HISTORICAL_FEATURES
PHASE4 = PHASE3 + TEAM_FEATURES + ["fixture_attack_mismatch", "fixture_defence_mismatch", "fixture_xg_environment"]
PARAMS = {"n_estimators": 30, "min_samples_leaf": 5, "random_state": 42, "n_jobs": -1}


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "minutes", "starts", "total_points", "goals_scored", "assists",
        "expected_goals", "expected_assists",
    ]
    agg = {c: "sum" for c in numeric if c in df.columns}
    agg["name"] = "first"
    agg["value"] = "first"
    agg["was_home"] = "first"
    agg["team"] = "first"
    agg["opponent_team"] = "first"
    weekly = df[df["season"] == "2025-26"].groupby(["element", "GW"], as_index=False).agg(agg)
    weekly["season"] = "2025-26"
    weekly = weekly.sort_values(["element", "GW"]).reset_index(drop=True)
    for col in [
        "total_points", "minutes", "expected_goals", "expected_assists",
    ]:
        for window in (3, 5, 10):
            weekly[f"previous_{window}_{col}"] = (
                weekly.groupby("element")[col]
                .transform(lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean())
            )
    weekly["previous_5_start_rate"] = weekly.groupby("element")["starts"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    weekly["previous_10_start_rate"] = weekly.groupby("element")["starts"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()
    )
    weekly["previous_5_minutes_rate"] = weekly.groupby("element")["minutes"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    ) / 90
    weekly["previous_10_minutes_rate"] = weekly.groupby("element")["minutes"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()
    ) / 90
    return weekly


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "mae": round(float(mean_absolute_error(actual, pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(actual, pred))), 4),
        "r2": round(float(r2_score(actual, pred)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("phase3_backtest.json"))
    args = parser.parse_args()

    combined = load_completed_seasons()
    weekly = aggregate_weekly(combined)
    prepared = prepare_historical_training(weekly)
    prepared = attach_team_features(prepared, build_historical_team_features())
    prepared = add_fixture_environment(prepared)

    rows = []
    for gw in range(11, 39):
        train = prepared[
            (prepared["GW"] < gw) & prepared["previous_3_total_points"].notna()
        ].copy()
        test = prepared[
            (prepared["GW"] == gw) & prepared["previous_3_total_points"].notna()
        ].copy()
        if len(train) < 500 or test.empty:
            continue

        y_train = train["total_points"].to_numpy()
        y_test = test["total_points"].to_numpy()

        for label, features in [("baseline", BASELINE), ("phase3", PHASE3), ("phase4", PHASE4)]:
            X = train[features].replace([np.inf, -np.inf], np.nan).fillna(0)
            Xt = test[features].replace([np.inf, -np.inf], np.nan).fillna(0)
            model = RandomForestRegressor(**PARAMS)
            model.fit(X, y_train)
            pred = np.clip(model.predict(Xt), 0, None)
            m = metrics(y_test, pred)
            rows.append({"gameweek": gw, "model": label, "rows": len(test), **m})

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No walk-forward gameweeks were evaluated")

    summary = (
        result.groupby("model")[["mae", "rmse", "r2"]]
        .mean()
        .round(4)
        .to_dict(orient="index")
    )
    output = {
        "evaluation": "walk-forward 2025/26, GW11-GW38",
        "models": summary,
        "gameweeks_evaluated": sorted(result["gameweek"].unique().tolist()),
        "detail": result.to_dict(orient="records"),
        "feature_count": {"baseline": len(BASELINE), "phase3": len(PHASE3), "phase4": len(PHASE4)},
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
