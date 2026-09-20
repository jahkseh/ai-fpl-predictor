"""Phase 4 team/opponent intelligence and fixture environment features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from historical_intelligence import TEAM_FEATURES, build_team_weekly_history, load_completed_seasons


FIXTURE_FEATURES = TEAM_FEATURES + [
    "fixture_attack_mismatch",
    "fixture_defence_mismatch",
    "fixture_xg_environment",
]


def build_historical_team_features() -> pd.DataFrame:
    history = load_completed_seasons()
    return build_team_weekly_history(history)


def attach_team_features(weekly: pd.DataFrame, team_history: pd.DataFrame) -> pd.DataFrame:
    out = weekly.copy()
    keys = ["season", "GW", "team", "opponent_team", "was_home"]
    for col in keys:
        if col not in out.columns:
            raise ValueError(f"Weekly training data missing {col}")
    lookup = team_history[keys + TEAM_FEATURES].drop_duplicates(keys)
    return out.merge(lookup, on=keys, how="left", validate="many_to_one")


def add_fixture_environment(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in TEAM_FEATURES:
        if col not in out.columns:
            out[col] = np.nan

    out["fixture_attack_mismatch"] = (
        out["team_xg_for_per90"] - out["opponent_xg_against_per90"]
    )
    out["fixture_defence_mismatch"] = (
        out["opponent_xg_for_per90"] - out["team_xg_against_per90"]
    )
    out["fixture_xg_environment"] = (
        out["team_xg_for_per90"] + out["opponent_xg_for_per90"]
    )
    return out


def add_current_fixture_team_features(
    target: pd.DataFrame,
    current: pd.DataFrame,
    gameweek: int,
) -> pd.DataFrame:
    """Attach current-season team rates using completed fixtures only."""
    required = {"team_code", "GW", "opponent_team_code", "expected_goals", "goals_scored"}
    missing = required - set(current.columns)
    if missing:
        raise ValueError(f"Current team history missing columns: {sorted(missing)}")

    completed = current[pd.to_numeric(current["GW"], errors="coerce") < gameweek].copy()
    completed["GW"] = pd.to_numeric(completed["GW"], errors="coerce")
    completed["team_code"] = pd.to_numeric(completed["team_code"], errors="coerce")
    completed["opponent_team_code"] = pd.to_numeric(completed["opponent_team_code"], errors="coerce")
    completed = completed.dropna(subset=["GW", "team_code", "opponent_team_code"])
    completed["team_code"] = completed["team_code"].astype(int)
    completed["opponent_team_code"] = completed["opponent_team_code"].astype(int)

    fixture = (
        completed.groupby(["GW", "team_code", "opponent_team_code"], as_index=False)
        .agg(xg_for=("expected_goals", "sum"), goals_for=("goals_scored", "sum"))
    )
    opponent = fixture[["GW", "team_code", "xg_for", "goals_for"]].rename(
        columns={"team_code": "opponent_team_code", "xg_for": "xg_against", "goals_for": "goals_against"}
    )
    fixture = fixture.merge(
        opponent, on=["GW", "opponent_team_code"], how="left", validate="many_to_one"
    )
    fixture = fixture.sort_values(["team_code", "GW"]).reset_index(drop=True)

    def rolling_prior(series: pd.Series) -> pd.Series:
        return series.shift(1).rolling(10, min_periods=1).mean()

    for src, dst in [
        ("xg_for", "team_xg_for_per90"),
        ("goals_for", "team_goals_for_per90"),
        ("xg_against", "team_xg_against_per90"),
        ("goals_against", "team_goals_against_per90"),
    ]:
        fixture[dst] = fixture.groupby("team_code")[src].transform(rolling_prior)

    latest = fixture.groupby("team_code").tail(1).copy()
    latest = latest.drop(columns=["GW", "opponent_team_code", "xg_for", "goals_for", "xg_against", "goals_against"])

    opponent_latest = fixture.groupby("opponent_team_code").tail(1).copy()
    opponent_latest = opponent_latest[
        ["opponent_team_code", "team_xg_for_per90", "team_goals_for_per90",
         "team_xg_against_per90", "team_goals_against_per90"]
    ].rename(columns={
        "team_xg_for_per90": "opponent_xg_for_per90",
        "team_goals_for_per90": "opponent_goals_for_per90",
        "team_xg_against_per90": "opponent_xg_against_per90",
        "team_goals_against_per90": "opponent_goals_against_per90",
    })

    out = target.copy().merge(
        latest, on="team_code", how="left", validate="many_to_one"
    ).merge(
        opponent_latest, on="opponent_team_code", how="left", validate="many_to_one"
    )

    for col in TEAM_FEATURES:
        if col not in out.columns:
            out[col] = np.nan
    return add_fixture_environment(out)


def phase4_feature_columns() -> list[str]:
    return list(FIXTURE_FEATURES)
