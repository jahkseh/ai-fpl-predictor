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
    teams: pd.DataFrame,
) -> pd.DataFrame:
    """Map current FPL team strength into the same feature family."""
    out = target.copy()
    team_lookup = teams.set_index("code")

    def val(code, col):
        try:
            return float(team_lookup.loc[code, col])
        except (KeyError, TypeError, ValueError):
            return np.nan

    home = out["was_home_gw"].astype(bool)
    out["team_xg_for_per90"] = np.where(
        home,
        [val(c, "strength_attack_home") for c in out["team_code"]],
        [val(c, "strength_attack_away") for c in out["team_code"]],
    )
    out["team_xg_against_per90"] = np.where(
        home,
        [val(c, "strength_defence_home") for c in out["team_code"]],
        [val(c, "strength_defence_away") for c in out["team_code"]],
    )
    out["opponent_xg_for_per90"] = np.where(
        home,
        [val(c, "strength_attack_away") for c in out["opponent_team_code"]],
        [val(c, "strength_attack_home") for c in out["opponent_team_code"]],
    )
    out["opponent_xg_against_per90"] = np.where(
        home,
        [val(c, "strength_defence_away") for c in out["opponent_team_code"]],
        [val(c, "strength_defence_home") for c in out["opponent_team_code"]],
    )

    # FPL strength values are ratings rather than xG/90. Keep them as a
    # separate environment signal; we do not pretend they are goals.
    out["team_goals_for_per90"] = out["team_xg_for_per90"]
    out["team_goals_against_per90"] = out["team_xg_against_per90"]
    out["opponent_goals_for_per90"] = out["opponent_xg_for_per90"]
    out["opponent_goals_against_per90"] = out["opponent_xg_against_per90"]

    for col in TEAM_FEATURES:
        if col not in out.columns:
            out[col] = np.nan
    return add_fixture_environment(out)


def phase4_feature_columns() -> list[str]:
    return list(FIXTURE_FEATURES)
