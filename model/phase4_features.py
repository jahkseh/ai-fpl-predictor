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
    """Use completed current-season team performance, not team ratings, for targets."""
    required = {"team_code", "GW", "opponent_team_code", "expected_goals",
                "expected_goals_conceded", "goals_scored"}
    missing = required - set(current.columns)
    if missing:
        raise ValueError(f"Current team history missing columns: {sorted(missing)}")

    completed = current[current["GW"] < gameweek].copy()
    team = (
        completed.groupby(["GW", "team_code"], as_index=False)
        .agg(
            xg_for=("expected_goals", "sum"),
            xga_against=("expected_goals_conceded", "sum"),
            goals_for=("goals_scored", "sum"),
        )
    )
    team = team.sort_values(["team_code", "GW"])
    for src, dst in [
        ("xg_for", "team_xg_for_per90"),
        ("goals_for", "team_goals_for_per90"),
        ("xga_against", "team_xg_against_per90"),
    ]:
        team[dst] = team.groupby("team_code")[src].transform(
            lambda s: s.shift(1).rolling(10, min_periods=1).mean()
        )
    team["team_goals_against_per90"] = np.nan

    opp = team[[
        "GW", "team_code", "team_xg_for_per90",
        "team_goals_for_per90", "team_xg_against_per90",
        "team_goals_against_per90",
    ]].rename(columns={
        "team_code": "opponent_team_code",
        "team_xg_for_per90": "opponent_xg_for_per90",
        "team_goals_for_per90": "opponent_goals_for_per90",
        "team_xg_against_per90": "opponent_xg_against_per90",
        "team_goals_against_per90": "opponent_goals_against_per90",
    })

    latest = team.groupby("team_code").tail(1).drop(columns=["GW", "xg_for", "xga_against", "goals_for"])
    out = target.copy()
    out = out.merge(
        latest, on="team_code", how="left", validate="many_to_one"
    )
    out = out.merge(
        opp.groupby("opponent_team_code").tail(1),
        on="opponent_team_code", how="left", validate="many_to_one"
    )
    for col in TEAM_FEATURES:
        if col not in out.columns:
            out[col] = np.nan
    return add_fixture_environment(out)



def phase4_feature_columns() -> list[str]:
    return list(FIXTURE_FEATURES)
