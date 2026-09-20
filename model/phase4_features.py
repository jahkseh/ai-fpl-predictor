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
    fixture_history: pd.DataFrame,
    gameweek: int,
) -> pd.DataFrame:
    """Attach current-season team rates using completed PL fixtures only."""
    required = {
        "gameweek", "home_team", "away_team", "home_score", "away_score",
        "home_expected_goals_xg", "away_expected_goals_xg", "tournament", "finished",
    }
    missing = required - set(fixture_history.columns)
    if missing:
        raise ValueError(f"Current fixture history missing columns: {sorted(missing)}")

    fixtures = fixture_history.copy()
    fixtures["gameweek"] = pd.to_numeric(fixtures["gameweek"], errors="coerce")
    fixtures = fixtures[
        (fixtures["tournament"] == "prem")
        & fixtures["finished"].fillna(False).astype(bool)
        & (fixtures["gameweek"] < gameweek)
    ].copy()
    for col in [
        "home_team", "away_team", "home_score", "away_score",
        "home_expected_goals_xg", "away_expected_goals_xg",
    ]:
        fixtures[col] = pd.to_numeric(fixtures[col], errors="coerce")
    fixtures = fixtures.dropna(
        subset=["gameweek", "home_team", "away_team", "home_score", "away_score",
                "home_expected_goals_xg", "away_expected_goals_xg"]
    )

    home = fixtures[["gameweek", "home_team", "away_team", "home_score",
                     "home_expected_goals_xg"]].rename(columns={
        "home_team": "team_code",
        "away_team": "opponent_team_code",
        "home_score": "goals_for",
        "home_expected_goals_xg": "xg_for",
    })
    home["was_home"] = 1

    away = fixtures[["gameweek", "home_team", "away_team", "away_score",
                     "away_expected_goals_xg"]].rename(columns={
        "away_team": "team_code",
        "home_team": "opponent_team_code",
        "away_score": "goals_for",
        "away_expected_goals_xg": "xg_for",
    })
    away["was_home"] = 0

    team_fixtures = pd.concat([home, away], ignore_index=True)
    team_fixtures["team_code"] = team_fixtures["team_code"].astype(int)
    team_fixtures["opponent_team_code"] = team_fixtures["opponent_team_code"].astype(int)

    opponent = team_fixtures[
        ["gameweek", "team_code", "goals_for", "xg_for"]
    ].rename(columns={
        "team_code": "opponent_team_code",
        "goals_for": "goals_against",
        "xg_for": "xg_against",
    })
    team_fixtures = team_fixtures.merge(
        opponent,
        on=["gameweek", "opponent_team_code"],
        how="left",
        validate="many_to_one",
    ).sort_values(["team_code", "gameweek"]).reset_index(drop=True)

    def rolling_prior(series: pd.Series) -> pd.Series:
        return series.shift(1).rolling(10, min_periods=1).mean()

    for src, dst in [
        ("xg_for", "team_xg_for_per90"),
        ("goals_for", "team_goals_for_per90"),
        ("xg_against", "team_xg_against_per90"),
        ("goals_against", "team_goals_against_per90"),
    ]:
        team_fixtures[dst] = team_fixtures.groupby("team_code")[src].transform(rolling_prior)

    for home_flag, label in [(1, "home"), (0, "away")]:
        subset = team_fixtures[team_fixtures["was_home"].eq(home_flag)].copy()
        for src, dst in [
            ("xg_for", f"team_{label}_xg_for_per90"),
            ("xg_against", f"team_{label}_xg_against_per90"),
        ]:
            subset[dst] = subset.groupby("team_code")[src].transform(rolling_prior)
            team_fixtures.loc[subset.index, dst] = subset[dst]

    latest = team_fixtures.groupby("team_code").tail(1).copy()
    latest = latest.drop(columns=[
        "gameweek", "opponent_team_code", "was_home",
        "xg_for", "goals_for", "xg_against", "goals_against",
    ])

    opponent_latest = latest.rename(columns={
        "team_code": "opponent_team_code",
        "team_xg_for_per90": "opponent_xg_for_per90",
        "team_goals_for_per90": "opponent_goals_for_per90",
        "team_xg_against_per90": "opponent_xg_against_per90",
        "team_goals_against_per90": "opponent_goals_against_per90",
    })[
        ["opponent_team_code", "opponent_xg_for_per90", "opponent_goals_for_per90",
         "opponent_xg_against_per90", "opponent_goals_against_per90"]
    ]

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
