"""Phase 3 feature engineering and model helpers.

This module deliberately separates:
1. player historical ability,
2. current form,
3. expected minutes,
4. workload/context.

Every training feature is constructed from information available before the
row's gameweek. It does not use future FPL points.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from historical_intelligence import (
    add_historical_priors,
    build_current_history_lookup,
    historical_feature_columns,
    load_completed_seasons,
    normalize_name,
)

HISTORICAL_FEATURES = historical_feature_columns()

MINUTES_FEATURES = [
    "previous_3_minutes", "previous_5_minutes", "previous_10_minutes",
    "previous_5_start_rate", "previous_10_start_rate",
    "previous_5_minutes_rate", "previous_10_minutes_rate",
    "historical_start_rate", "club_start_rate",
]

WORKLOAD_FEATURES = [
    "matches_last_3_gw", "minutes_last_3_gw",
    "matches_last_5_gw", "minutes_last_5_gw",
]

MODEL_FEATURES = HISTORICAL_FEATURES + [
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

PHASE3_MODEL_PARAMS = {
    "n_estimators": 250,
    "min_samples_leaf": 5,
    "random_state": 42,
    "n_jobs": -1,
}


def prepare_historical_training(weekly: pd.DataFrame) -> pd.DataFrame:
    """Attach 2025/26 prior-season and club-history features to weekly rows."""
    combined = add_historical_priors(load_completed_seasons())
    hist = combined[combined["season"] == "2025-26"].copy()
    hist["player_key"] = hist["name"].map(normalize_name)

    # Match the validated weekly aggregation to the detailed historical rows.
    # The historical priors are already point-in-time at each GW.
    prior = hist[
        ["player_key", "GW"] + HISTORICAL_FEATURES
    ].drop_duplicates(["player_key", "GW"])

    out = weekly.copy()
    out["player_key"] = out["name"].map(normalize_name)
    out = out.merge(
        prior, on=["player_key", "GW"], how="left", validate="one_to_one"
    )
    return out


def add_workload_features(frame: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Add recent match/minute load using completed FPL gameweeks only."""
    out = frame.sort_values([id_col, "GW"]).copy()
    out["matches_last_3_gw"] = (
        out.groupby(id_col)["GW"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).count())
    )
    out["minutes_last_3_gw"] = (
        out.groupby(id_col)["minutes"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())
    )
    out["matches_last_5_gw"] = (
        out.groupby(id_col)["GW"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).count())
    )
    out["minutes_last_5_gw"] = (
        out.groupby(id_col)["minutes"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum())
    )
    return out


def add_target_historical_features(
    target: pd.DataFrame,
    historical: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the latest completed historical profile to current-season players."""
    lookup = build_current_history_lookup(historical)
    out = target.copy()
    name_col = "web_name" if "web_name" in out.columns else "name"
    out["player_key"] = out[name_col].map(normalize_name)
    out = out.merge(
        lookup,
        on="player_key",
        how="left",
        validate="many_to_one",
        suffixes=("", "_history"),
    )
    return out


def fit_expected_minutes_models(
    training: pd.DataFrame,
    target: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Estimate start probability, 60+ probability and expected minutes."""
    features = [c for c in MINUTES_FEATURES if c in training.columns and c in target.columns]
    train = training.dropna(subset=features).copy()
    pred = target.copy()

    if not features or len(train) < 100:
        pred["start_probability"] = np.nan
        pred["sixty_probability"] = np.nan
        pred["expected_minutes"] = pred["previous_10_minutes"].fillna(
            pred.get("previous_5_minutes", 0)
        )
        return pred, {"enabled": False, "reason": "insufficient training rows"}

    X = train[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    Xt = pred[features].replace([np.inf, -np.inf], np.nan).fillna(0)

    starts = RandomForestClassifier(**PHASE3_MODEL_PARAMS)
    sixty = RandomForestClassifier(**PHASE3_MODEL_PARAMS)
    minutes = RandomForestRegressor(**PHASE3_MODEL_PARAMS)

    starts.fit(X, (train["starts"] > 0).astype(int))
    sixty.fit(X, (train["minutes"] >= 60).astype(int))
    minutes.fit(X, train["minutes"].clip(0, 90))

    pred["start_probability"] = starts.predict_proba(Xt)[:, 1]
    pred["sixty_probability"] = sixty.predict_proba(Xt)[:, 1]
    pred["expected_minutes"] = np.clip(minutes.predict(Xt), 0, 90)

    return pred, {
        "enabled": True,
        "training_rows": int(len(train)),
        "features": features,
    }


def add_phase3_defaults(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill absent historical features conservatively; never fabricate ability."""
    out = frame.copy()
    for col in HISTORICAL_FEATURES:
        if col not in out.columns:
            out[col] = np.nan
    return out
