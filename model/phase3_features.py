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

# FPL's ICT/BPS signals add information beyond raw points and xG/xA.
# They are all lagged and converted to per-90 rates so playing time does not
# dominate the signal. These fields are present across the historical FPL
# gameweek archive and in the current FPL-Core-Insights feed.
PERFORMANCE_SOURCE_COLUMNS = [
    "ict_index", "influence", "creativity", "threat", "bps", "bonus",
]
PERFORMANCE_FEATURES = [
    f"previous_{window}_{metric}_per90"
    for metric in PERFORMANCE_SOURCE_COLUMNS
    for window in (3, 5, 10)
]

MODEL_FEATURES = HISTORICAL_FEATURES + PERFORMANCE_FEATURES + [
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



def add_performance_features(frame: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Add leakage-safe rolling ICT/BPS/bonus performance signals."""
    sort_keys = [id_col, "season", "GW"] if "season" in frame.columns else [id_col, "GW"]
    group_keys = [id_col, "season"] if "season" in frame.columns else [id_col]
    out = frame.sort_values(sort_keys).copy()

    for source in PERFORMANCE_SOURCE_COLUMNS:
        if source not in out.columns:
            out[source] = 0.0
        values = pd.to_numeric(out[source], errors="coerce").fillna(0.0)
        minutes = pd.to_numeric(out["minutes"], errors="coerce").fillna(0.0)
        per90 = values.div(minutes.replace(0, np.nan)).mul(90)
        # A zero-minute row contains no performance signal. Treat it as
        # missing for the rolling calculation rather than manufacturing a
        # zero-performance appearance.
        per90 = per90.where(minutes > 0)
        out[f"_tmp_{source}_per90"] = per90
        for window in (3, 5, 10):
            out[f"previous_{window}_{source}_per90"] = (
                out.groupby(group_keys)[f"_tmp_{source}_per90"]
                .transform(
                    lambda x, w=window:
                    x.shift(1).rolling(w, min_periods=1).mean()
                )
            )

    return out.drop(columns=[c for c in out.columns if c.startswith("_tmp_")])

def prepare_historical_training(weekly: pd.DataFrame) -> pd.DataFrame:
    """Attach leakage-safe historical priors to every completed season."""
    combined = add_historical_priors(load_completed_seasons())

    if "season" not in weekly.columns or "element" not in weekly.columns:
        raise ValueError("Weekly historical training data must include season and element")

    # Use the season-local FPL element ID for the point-in-time join. Names are
    # not guaranteed to be unique, so they are not safe as the training key.
    prior = combined[
        ["season", "element", "GW"] + HISTORICAL_FEATURES
    ].drop_duplicates(["season", "element", "GW"])

    out = weekly.copy()
    out = out.merge(
        prior,
        on=["season", "element", "GW"],
        how="left",
        validate="one_to_one",
    )
    # Keep a normalized identity for downstream workload grouping and target
    # joins after the season-local historical join.
    out["player_key"] = out["name"].map(normalize_name)
    return out
def add_workload_features(frame: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Add recent match/minute load without crossing a season boundary."""
    sort_keys = [id_col, "season", "GW"] if "season" in frame.columns else [id_col, "GW"]
    group_keys = [id_col, "season"] if "season" in frame.columns else [id_col]
    out = frame.sort_values(sort_keys).copy()

    out["matches_last_3_gw"] = (
        out.groupby(group_keys)["GW"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).count())
    )
    out["minutes_last_3_gw"] = (
        out.groupby(group_keys)["minutes"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())
    )
    out["matches_last_5_gw"] = (
        out.groupby(group_keys)["GW"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).count())
    )
    out["minutes_last_5_gw"] = (
        out.groupby(group_keys)["minutes"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum())
    )
    return out
def add_target_historical_features(
    target: pd.DataFrame,
    historical: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the latest completed historical profile to current-season players.

    Current FPL web_name values are often shortened (for example, "Rice"),
    while the historical archive may store the player's full name
    (for example, "Declan Rice"). Match on the current first/second name
    combination when it is available, with web_name as a fallback. This keeps
    historical ability and club-history features populated after transfers or
    display-name changes.
    """
    lookup = build_current_history_lookup(historical)
    out = target.copy()
    lookup_keys = set(lookup["player_key"])

    if {"first_name", "second_name"} <= set(out.columns):
        full_name = (
            out["first_name"].fillna("").astype(str).str.strip()
            + " "
            + out["second_name"].fillna("").astype(str).str.strip()
        ).str.strip()
        out["player_key"] = full_name.map(normalize_name)
        web_keys = (
            out["web_name"].map(normalize_name)
            if "web_name" in out.columns
            else pd.Series(index=out.index, dtype=object)
        )
    else:
        name_col = "web_name" if "web_name" in out.columns else "name"
        out["player_key"] = out[name_col].map(normalize_name)
        web_keys = pd.Series(index=out.index, dtype=object)

    # If the full-name key is unavailable, use a unique surname match. This is
    # only a fallback because web_name is not guaranteed to be globally unique.
    surname_candidates = {}
    for key in lookup_keys:
        parts = key.split()
        if parts:
            surname_candidates.setdefault(parts[-1], []).append(key)

    resolved = []
    for idx, key in out["player_key"].items():
        if key in lookup_keys:
            resolved.append(key)
            continue
        web_key = web_keys.loc[idx] if idx in web_keys.index else ""
        candidates = surname_candidates.get(web_key, [])
        resolved.append(candidates[0] if len(candidates) == 1 else key)

    out["player_key"] = resolved
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
