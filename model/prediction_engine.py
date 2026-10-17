"""Prediction distribution and presentation helpers for the FPL model.

The Random Forest prediction is kept as the point estimate. Tree-level
predictions are also used to estimate model spread; this is uncertainty in the
model, not a probability that the player will score a particular total.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def safe_json_float(value):
    value = float(value)
    return round(value, 6) if np.isfinite(value) else None


def add_prediction_distribution(model, frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Add point prediction, model spread, and a conservative interval."""
    out = frame.copy()
    X = out[features]
    if X.isna().any().any():
        raise ValueError("Prediction matrix contains missing values")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Prediction matrix contains non-finite values")

    tree_predictions = np.vstack([tree.predict(X) for tree in model.estimators_]).T
    point = tree_predictions.mean(axis=1)
    spread = tree_predictions.std(axis=1, ddof=1)
    lower = np.percentile(tree_predictions, 10, axis=1)
    upper = np.percentile(tree_predictions, 90, axis=1)

    point = np.clip(point, 0, None)
    # Percentiles of a skewed tree ensemble can occasionally fail to bracket
    # its mean. Keep the displayed interval valid and ensure it contains the
    # point estimate, including for players whose raw estimate is below zero.
    lower = np.minimum(lower, point)
    upper = np.maximum(upper, point)

    out["prediction"] = point
    out["prediction_std"] = np.maximum(spread, 0)
    out["prediction_p10"] = np.clip(lower, 0, None)
    out["prediction_p90"] = np.clip(upper, 0, None)
    out["prediction_range"] = out["prediction_p90"] - out["prediction_p10"]
    out["confidence"] = np.select(
        [
            out["prediction_std"] <= 1.0,
            out["prediction_std"] <= 1.75,
        ],
        ["high", "medium"],
        default="low",
    )
    return out


def player_prediction_payload(row: pd.Series) -> dict:
    """Return JSON-safe model context without changing legacy fields."""
    return {
        "prediction_std": round(float(row["prediction_std"]), 6),
        "prediction_p10": round(float(row["prediction_p10"]), 6),
        "prediction_p90": round(float(row["prediction_p90"]), 6),
        "prediction_range": round(float(row["prediction_range"]), 6),
        "confidence": str(row["confidence"]),
        "fixture_difficulty": int(row["fixture_difficulty"]) if pd.notna(row.get("fixture_difficulty", np.nan)) else None,
        "expected_minutes": safe_json_float(row.get("expected_minutes", np.nan)),
        "start_probability": safe_json_float(row.get("start_probability", np.nan)),
        "sixty_probability": safe_json_float(row.get("sixty_probability", np.nan)),
        "fixture_context": {
            "opponent": str(row["opponent"]),
            "is_home": bool(row["was_home_gw"]),
            "team_strength": safe_json_float(row["team_strength"]),
            "opponent_strength": safe_json_float(row["opponent_strength"]),
            "home_elo": safe_json_float(row["home_elo"]),
            "away_elo": safe_json_float(row["away_elo"]),
            "team_xg_for_per90": safe_json_float(row.get("team_xg_for_per90", np.nan)),
            "team_xg_against_per90": safe_json_float(row.get("team_xg_against_per90", np.nan)),
            "opponent_xg_for_per90": safe_json_float(row.get("opponent_xg_for_per90", np.nan)),
            "opponent_xg_against_per90": safe_json_float(row.get("opponent_xg_against_per90", np.nan)),
            "fixture_attack_mismatch": safe_json_float(row.get("fixture_attack_mismatch", np.nan)),
            "fixture_defence_mismatch": safe_json_float(row.get("fixture_defence_mismatch", np.nan)),
            "fixture_xg_environment": safe_json_float(row.get("fixture_xg_environment", np.nan)),
        },
        # Descriptive model inputs exposed to the player page. These do not
        # claim causal feature attribution.
        "model_inputs": {
            "historical_ability": {
                "points_per_90": safe_json_float(row.get("historical_points_per_90", np.nan)),
                "xg_per_90": safe_json_float(row.get("historical_xg_per_90", np.nan)),
                "xa_per_90": safe_json_float(row.get("historical_xa_per_90", np.nan)),
                "goals_per_90": safe_json_float(row.get("historical_goals_per_90", np.nan)),
                "assists_per_90": safe_json_float(row.get("historical_assists_per_90", np.nan)),
                "start_rate": safe_json_float(row.get("historical_start_rate", np.nan)),
            },
            "current_form": {
                "points_per_90": safe_json_float(
                    row.get("previous_3_total_points", np.nan)
                    / (row.get("previous_3_minutes", np.nan) / 90)
                    if pd.notna(row.get("previous_3_total_points", np.nan))
                    and pd.notna(row.get("previous_3_minutes", np.nan))
                    and float(row.get("previous_3_minutes", 0)) > 0
                    else np.nan
                ),
                "recent_3_points": safe_json_float(row.get("previous_3_total_points", np.nan)),
                "recent_3_minutes": safe_json_float(row.get("previous_3_minutes", np.nan)),
                "recent_3_xg": safe_json_float(row.get("previous_3_expected_goals", np.nan)),
                "recent_3_xa": safe_json_float(row.get("previous_3_expected_assists", np.nan)),
                "ict_index_per_90": safe_json_float(row.get("previous_5_ict_index_per90", np.nan)),
                "influence_per_90": safe_json_float(row.get("previous_5_influence_per90", np.nan)),
                "creativity_per_90": safe_json_float(row.get("previous_5_creativity_per90", np.nan)),
                "threat_per_90": safe_json_float(row.get("previous_5_threat_per90", np.nan)),
                "bps_per_90": safe_json_float(row.get("previous_5_bps_per90", np.nan)),
                "bonus_per_90": safe_json_float(row.get("previous_5_bonus_per90", np.nan)),
                "xgi_per_90": safe_json_float(
                    (row.get("previous_5_expected_goals", np.nan) + row.get("previous_5_expected_assists", np.nan))
                    / (row.get("previous_5_minutes", np.nan) / 90)
                    if pd.notna(row.get("previous_5_expected_goals", np.nan))
                    and pd.notna(row.get("previous_5_expected_assists", np.nan))
                    and pd.notna(row.get("previous_5_minutes", np.nan))
                    and float(row.get("previous_5_minutes", 0)) > 0
                    else np.nan
                ),
            },
            "current_club": {
                "points_per_90": safe_json_float(row.get("club_points_per_90", np.nan)),
                "xg_per_90": safe_json_float(row.get("club_xg_per_90", np.nan)),
                "xa_per_90": safe_json_float(row.get("club_xa_per_90", np.nan)),
                "start_rate": safe_json_float(row.get("club_start_rate", np.nan)),
            },
            "team_attack": {
                "xg_for_per_90": safe_json_float(row.get("team_xg_for_per90", np.nan)),
                "goals_for_per_90": safe_json_float(row.get("team_goals_for_per90", np.nan)),
            },
            "opponent_defence": {
                "xg_against_per_90": safe_json_float(row.get("opponent_xg_against_per90", np.nan)),
                "goals_against_per_90": safe_json_float(row.get("opponent_goals_against_per90", np.nan)),
            },
            "expected_minutes": safe_json_float(row.get("expected_minutes", np.nan)),
            "home_away": "home" if bool(row["was_home_gw"]) else "away",
            "fixture_environment": safe_json_float(row.get("fixture_xg_environment", np.nan)),
        },
    }
