#!/usr/bin/env python3
"""Automated FPL prediction pipeline based on the validated Colab model."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from prediction_engine import add_prediction_distribution, player_prediction_payload
from phase3_features import (
    HISTORICAL_FEATURES,
    add_target_historical_features,
    add_workload_features,
    fit_expected_minutes_models,
    prepare_historical_training,
)
from historical_intelligence import add_historical_priors, load_completed_seasons
from phase4_features import (
    phase4_feature_columns,
    build_historical_team_features,
    attach_team_features,
    add_current_fixture_team_features,
    add_fixture_environment,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


HISTORICAL_URL = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
    "master/data/2025-26/gws/merged_gw.csv"
)
CURRENT_BASE = (
    "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/"
    "data/2026-2027/By%20Gameweek/GW{gw}/"
)

BASE_PLAYER_FEATURES = [
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
PHASE4_FEATURES = phase4_feature_columns()
PLAYER_FEATURES = BASE_PLAYER_FEATURES + HISTORICAL_FEATURES + PHASE4_FEATURES

MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 5,
    "random_state": 42,
    "n_jobs": -1,
}

PLAYER_NAME_OVERRIDES = {"B.Fernandes": "B. Fernandes"}

ROLLING_COLUMNS = [
    "total_points", "minutes", "goals_scored", "assists",
    "expected_goals", "expected_assists",
]


@dataclass
class PipelineResult:
    gameweek: int
    eligible_players: int
    predictions: pd.DataFrame
    diagnostics: dict


def read_csv(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    if df.empty:
        raise ValueError(f"Downloaded empty CSV: {url}")
    return df


def read_csv_optional(url: str) -> pd.DataFrame | None:
    try:
        return read_csv(url)
    except (ValueError, pd.errors.EmptyDataError):
        return None


def current_url(gameweek: int, filename: str) -> str:
    return CURRENT_BASE.format(gw=gameweek) + filename


def load_historical_data() -> pd.DataFrame:
    df = read_csv(HISTORICAL_URL)
    required = {
        "element", "GW", "name", "value", "was_home", "minutes", "starts",
        "total_points", "goals_scored", "assists", "expected_goals",
        "expected_assists",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Historical data missing columns: {sorted(missing)}")
    return df


def build_weekly_historical(df: pd.DataFrame) -> pd.DataFrame:
    # The validated Colab model covers GW1-GW38 only. Ignore later upstream rows.
    df = df[df["GW"].between(1, 38)].copy()

    sum_columns = [
        "minutes", "starts", "total_points", "goals_scored", "assists",
        "expected_goals", "expected_assists", "clean_sheets", "goals_conceded",
        "saves", "bonus", "bps", "yellow_cards", "red_cards", "own_goals",
        "penalties_saved", "penalties_missed",
    ]
    sum_columns = [c for c in sum_columns if c in df.columns]
    first_columns = [c for c in ["name", "value", "was_home", "team", "opponent_team"] if c in df.columns]

    aggregation = {c: "sum" for c in sum_columns}
    aggregation.update({c: "first" for c in first_columns})

    weekly = (
        df.groupby(["element", "GW"], as_index=False)
        .agg(aggregation)
        .sort_values(["element", "GW"])
        .reset_index(drop=True)
    )

    if weekly.duplicated(["element", "GW"]).any():
        raise AssertionError("Historical weekly data contains duplicate player/GW rows")

    for col in ROLLING_COLUMNS:
        for window in (3, 5, 10):
            weekly[f"previous_{window}_{col}"] = (
                weekly.groupby("element")[col]
                .transform(
                    lambda x, w=window:
                    x.shift(1).rolling(w, min_periods=1).mean()
                )
            )

    weekly["previous_5_start_rate"] = (
        weekly.groupby("element")["starts"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    weekly["previous_10_start_rate"] = (
        weekly.groupby("element")["starts"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )
    weekly["previous_5_minutes_rate"] = (
        weekly.groupby("element")["minutes"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()) / 90
    )
    weekly["previous_10_minutes_rate"] = (
        weekly.groupby("element")["minutes"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean()) / 90
    )

    return weekly


def load_current_data(
    gameweek: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    for gw in range(1, gameweek + 1):
        frame = read_csv_optional(current_url(gw, "player_gameweek_stats.csv"))
        if frame is not None:
            frames.append(frame.assign(GW=gw))
    if not frames:
        raise ValueError("No current player gameweek data could be loaded")
    current = pd.concat(frames, ignore_index=True)

    if not (current["GW"] == gameweek).any():
        # Some upstream feeds publish next-GW metadata before the gameweek
        # stat file is populated. Use the latest player snapshot as a target
        # shell and reset performance fields to prevent target leakage.
        latest = current.sort_values(["id", "GW"]).groupby("id").tail(1).copy()
        zero_columns = [
            "total_points", "minutes", "starts", "goals_scored", "assists",
            "clean_sheets", "goals_conceded", "own_goals", "penalties_saved",
            "penalties_missed", "yellow_cards", "red_cards", "saves", "bonus",
            "bps", "transfers_in", "transfers_out", "expected_goals",
            "expected_assists", "expected_goal_involvements",
            "expected_goals_conceded", "influence", "creativity", "threat",
            "ict_index", "tackles", "clearances_blocks_interceptions",
            "recoveries", "defensive_contribution",
        ]
        for col in zero_columns:
            if col in latest.columns:
                latest[col] = 0
        latest["GW"] = gameweek
        current = pd.concat([current, latest], ignore_index=True)

    players = read_csv(current_url(gameweek, "players.csv"))
    teams = read_csv(current_url(gameweek, "teams.csv"))

    required = {"id", "total_points", "minutes", "starts", "now_cost"}
    missing = required - set(current.columns)
    if missing:
        raise ValueError(f"Current player data missing columns: {sorted(missing)}")

    return current, players, teams


def enrich_current_players(
    current: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
) -> pd.DataFrame:
    player_lookup = (
        players[["player_id", "team_code", "position"]]
        .drop_duplicates("player_id")
    )
    team_lookup = (
        teams[[
            "code", "id", "name", "short_name", "strength",
            "strength_attack_home", "strength_attack_away",
            "strength_defence_home", "strength_defence_away", "elo",
        ]]
        .drop_duplicates("code")
    )

    enriched = current.merge(
        player_lookup,
        left_on="id",
        right_on="player_id",
        how="left",
        validate="many_to_one",
    )
    enriched = enriched.merge(
        team_lookup,
        left_on="team_code",
        right_on="code",
        how="left",
        suffixes=("", "_team"),
        validate="many_to_one",
    )

    if len(enriched) != len(current):
        raise AssertionError("Player enrichment changed row count")

    for col in ["team_code", "position", "web_name", "name", "short_name"]:
        if col in enriched.columns and enriched[col].isna().any():
            raise ValueError(f"Missing player/team mapping: {col}")

    return enriched


def add_current_rolling_features(enriched: pd.DataFrame) -> pd.DataFrame:
    enriched = enriched.sort_values(["id", "GW"]).copy()

    for col in ROLLING_COLUMNS:
        if col not in enriched.columns:
            raise ValueError(f"Missing rolling source column: {col}")
        for window in (3, 5, 10):
            enriched[f"previous_{window}_{col}"] = (
                enriched.groupby("id")[col]
                .transform(
                    lambda x, w=window:
                    x.shift(1).rolling(w, min_periods=1).mean()
                )
            )

    enriched["previous_5_start_rate"] = (
        enriched.groupby("id")["starts"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    enriched["previous_10_start_rate"] = (
        enriched.groupby("id")["starts"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )
    enriched["previous_5_minutes_rate"] = (
        enriched.groupby("id")["minutes"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()) / 90
    )
    enriched["previous_10_minutes_rate"] = (
        enriched.groupby("id")["minutes"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean()) / 90
    )

    return enriched


def validate_no_leakage(enriched: pd.DataFrame, gameweek: int) -> None:
    target = enriched[enriched["GW"] == gameweek]
    if target.empty:
        raise AssertionError(f"No target-gameweek rows for GW{gameweek}")

    first_rows = enriched.groupby("id").head(1)
    previous = [c for c in BASE_PLAYER_FEATURES if c.startswith("previous_")]
    previous_present = [c for c in previous if c in first_rows.columns]
    if previous_present and first_rows[previous_present].notna().any(axis=1).any():
        raise AssertionError("Rolling features contain data on a player's first row")


def load_fixtures(gameweek: int) -> pd.DataFrame:
    fixtures = read_csv(current_url(gameweek, "fixtures.csv"))
    required = {"tournament", "gameweek", "match_id", "home_team", "away_team"}
    missing = required - set(fixtures.columns)
    if missing:
        raise ValueError(f"Fixture data missing columns: {sorted(missing)}")

    fixtures = fixtures[
        (fixtures["tournament"] == "prem")
        & (fixtures["gameweek"] == gameweek)
    ].copy()

    if fixtures.empty:
        raise ValueError(f"No Premier League fixtures found for GW{gameweek}")

    return fixtures


FIXTURE_NAME_TO_TEAM = {
    "everton": "Everton",
    "ipswich-town": "Ipswich Town",
    "nottingham-forest": "Nott'm Forest",
    "coventry-city": "Coventry City",
    "leeds-united": "Leeds",
    "crystal-palace": "Crystal Palace",
    "manchester-city": "Man City",
    "sunderland": "Sunderland",
    "afc-bournemouth": "Bournemouth",
    "liverpool": "Liverpool",
    "fulham": "Fulham",
    "manchester-united": "Man Utd",
    "brighton-hove-albion": "Brighton",
    "arsenal": "Arsenal",
    "tottenham-hotspur": "Spurs",
    "aston-villa": "Aston Villa",
    "newcastle-united": "Newcastle",
    "hull-city": "Hull City",
    "brentford": "Brentford",
    "chelsea": "Chelsea",
}


def build_fixture_lookup(
    fixtures: pd.DataFrame,
    teams: pd.DataFrame,
) -> pd.DataFrame:
    team_name_to_code = (
        teams[["name", "code"]]
        .drop_duplicates("name")
        .set_index("name")["code"]
        .to_dict()
    )

    rows = []
    for _, fixture in fixtures.iterrows():
        match_id = str(fixture["match_id"]).replace("26-27-prem-", "")
        parts = match_id.split("-vs-")
        if len(parts) != 2:
            raise ValueError(f"Unexpected match_id: {fixture['match_id']}")

        home = FIXTURE_NAME_TO_TEAM.get(parts[0])
        away = FIXTURE_NAME_TO_TEAM.get(parts[1])
        if home is None or away is None:
            raise ValueError(f"Unknown fixture slug: {fixture['match_id']}")
        if home not in team_name_to_code or away not in team_name_to_code:
            raise ValueError(f"Fixture team missing from teams.csv: {home}/{away}")

        home_code = team_name_to_code[home]
        away_code = team_name_to_code[away]
        team_rows = teams.set_index("code")
        rows.extend([
            {
                "team_code": home_code,
                "opponent_team_code": away_code,
                "opponent": away,
                "fixture_home_team": home,
                "fixture_away_team": away,
                "match_id": fixture["match_id"],
                "was_home_gw": 1,
                "home_strength": team_rows.loc[home_code, "strength"],
                "away_strength": team_rows.loc[away_code, "strength"],
                "home_elo": team_rows.loc[home_code, "elo"],
                "away_elo": team_rows.loc[away_code, "elo"],
            },
            {
                "team_code": away_code,
                "opponent_team_code": home_code,
                "opponent": home,
                "fixture_home_team": home,
                "fixture_away_team": away,
                "match_id": fixture["match_id"],
                "was_home_gw": 0,
                "home_strength": team_rows.loc[home_code, "strength"],
                "away_strength": team_rows.loc[away_code, "strength"],
                "home_elo": team_rows.loc[home_code, "elo"],
                "away_elo": team_rows.loc[away_code, "elo"],
            },
        ])

    lookup = pd.DataFrame(rows)
    if lookup["team_code"].duplicated().any():
        raise AssertionError("A team appears in more than one target fixture")
    if len(lookup) != 20:
        raise AssertionError(f"Expected 20 team fixture mappings, got {len(lookup)}")
    return lookup


def train_validated_model(weekly: pd.DataFrame) -> RandomForestRegressor:
    train = prepare_historical_training(weekly)
    train = add_workload_features(train, "player_key")
    train = attach_team_features(train, build_historical_team_features())
    train = add_fixture_environment(train)
    train = train[
        train["GW"].between(2, 32)
        & train["previous_3_total_points"].notna()
    ].copy()

    X = train[PLAYER_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = train["total_points"]

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Historical training features contain non-finite values")

    model = RandomForestRegressor(**MODEL_PARAMS)
    model.fit(X, y)
    return model


def historical_holdout_diagnostics(weekly: pd.DataFrame) -> dict:
    prepared = add_workload_features(prepare_historical_training(weekly), "player_key")
    prepared = attach_team_features(prepared, build_historical_team_features())
    prepared = add_fixture_environment(prepared)
    train = prepared[
        prepared["GW"].between(2, 32)
        & prepared["previous_3_total_points"].notna()
    ]
    test = prepared[
        prepared["GW"].between(33, 38)
        & prepared["previous_3_total_points"].notna()
    ].copy()

    X_train = train[PLAYER_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train["total_points"]
    X_test = test[PLAYER_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_test = test["total_points"]

    model = RandomForestRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = float(r2_score(y_test, pred))

    baseline_pred = test["previous_3_total_points"].to_numpy()
    baseline = {
        "mae": float(mean_absolute_error(y_test, baseline_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, baseline_pred))),
        "r2": float(r2_score(y_test, baseline_pred)),
    }

    expected = {"mae": 1.0131, "rmse": 1.9953, "r2": 0.3152}

    importance = (
        pd.Series(model.feature_importances_, index=PLAYER_FEATURES)
        .sort_values(ascending=False)
    )

    return {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "expected_mae": expected["mae"],
        "expected_rmse": expected["rmse"],
        "expected_r2": expected["r2"],
        "recent_3_baseline": {
            "mae": round(baseline["mae"], 4),
            "rmse": round(baseline["rmse"], 4),
            "r2": round(baseline["r2"], 4),
        },
        "feature_importance": {
            str(k): round(float(v), 6) for k, v in importance.items()
        },
    }


def generate_target_predictions(
    enriched: pd.DataFrame,
    teams: pd.DataFrame,
    weekly_historical: pd.DataFrame,
    gameweek: int,
) -> tuple[pd.DataFrame, int]:
    target = enriched[enriched["GW"] == gameweek].copy()
    target["value"] = target["now_cost"] * 10

    fixtures = load_fixtures(gameweek)
    lookup = build_fixture_lookup(fixtures, teams)
    target = target.merge(
        lookup, on="team_code", how="left", validate="many_to_one"
    )

    if target["match_id"].isna().any():
        raise ValueError("Some target players could not be mapped to a fixture")

    target["was_home"] = target["was_home_gw"].astype(int)

    # Phase 3 historical intelligence: preserve a player's established level
    # and club-specific history after transfers (e.g. Mbeumo at Brentford).
    combined_history = add_historical_priors(load_completed_seasons())
    target = add_target_historical_features(target, combined_history)
    target = add_current_fixture_team_features(target, current, gameweek)
    eligible = target.dropna(subset=["previous_3_total_points"]).copy()

    for col in PLAYER_FEATURES:
        if col not in eligible.columns:
            eligible[col] = np.nan
    eligible[PLAYER_FEATURES] = (
        eligible[PLAYER_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    )

    model = train_validated_model(weekly_historical)
    eligible = add_prediction_distribution(model, eligible, PLAYER_FEATURES)

    # Separate expected-minutes model. It is intentionally kept independent
    # from the points target so minutes are never allowed to leak into it.
    minutes_training = add_workload_features(
        prepare_historical_training(weekly_historical), "player_key"
    )
    eligible, minutes_diagnostics = fit_expected_minutes_models(
        minutes_training, eligible
    )
    eligible["team_strength"] = np.where(
        eligible["was_home_gw"].astype(bool),
        eligible["home_strength"],
        eligible["away_strength"],
    )
    eligible["opponent_strength"] = np.where(
        eligible["was_home_gw"].astype(bool),
        eligible["away_strength"],
        eligible["home_strength"],
    )
    eligible["fixture"] = (
        eligible["fixture_home_team"] + " vs " + eligible["fixture_away_team"]
    )
    eligible["phase3_minutes_model"] = minutes_diagnostics.get("enabled", False)

    eligible = eligible.sort_values(
        ["fixture", "prediction"], ascending=[True, False]
    )
    eligible["rank"] = eligible.groupby("fixture").cumcount() + 1

    return eligible[eligible["rank"] <= 3].copy(), len(eligible)


def build_predictions_json(top3: pd.DataFrame, gameweek: int) -> dict:
    predictions = []
    for (home, away), group in top3.groupby(
        ["fixture_home_team", "fixture_away_team"], sort=True
    ):
        players = [
            {
                "rank": int(row["rank"]),
                "name": PLAYER_NAME_OVERRIDES.get(str(row["web_name"]), str(row["web_name"])),
                "prediction": round(float(row["prediction"]), 6),
                **player_prediction_payload(row),
            }
            for _, row in group.sort_values("rank").iterrows()
        ]
        if len(players) != 3:
            raise AssertionError(f"Fixture {home} vs {away} does not have 3 predictions")
        predictions.append({"home": home, "away": away, "players": players})

    if len(predictions) != 10:
        raise AssertionError(f"Expected 10 fixtures, got {len(predictions)}")

    return {"gameweek": gameweek, "predictions": predictions}


def compare_with_golden(output: dict, golden_path: Path | None) -> dict:
    if golden_path is None:
        return {"golden_match": None}

    golden = json.loads(golden_path.read_text(encoding="utf-8"))

    def flatten(data: dict) -> dict:
        return {
            (f["home"], f["away"], p["name"]): float(p["prediction"])
            for f in data["predictions"]
            for p in f["players"]
        }

    actual = flatten(output)
    expected = flatten(golden)
    shared = actual.keys() & expected.keys()
    max_diff = max(
        (abs(actual[k] - expected[k]) for k in shared),
        default=math.inf,
    )

    result = {
        "golden_keys_equal": set(actual) == set(expected),
        "golden_max_abs_difference": round(max_diff, 9),
        "golden_prediction_count": len(expected),
        "golden_missing_keys": sorted(set(expected) - set(actual)),
        "golden_extra_keys": sorted(set(actual) - set(expected)),
    }
    result["golden_match"] = (
        result["golden_keys_equal"] and max_diff <= 1e-6
    )
    return result


def run(gameweek: int, golden_path: Path | None = None) -> PipelineResult:
    historical = load_historical_data()
    weekly = build_weekly_historical(historical)

    current, players, teams = load_current_data(gameweek)
    enriched = add_current_rolling_features(
        enrich_current_players(current, players, teams)
    )
    validate_no_leakage(enriched, gameweek)

    top3, eligible = generate_target_predictions(
        enriched, teams, weekly, gameweek
    )
    output = build_predictions_json(top3, gameweek)

    checks = compare_with_golden(output, golden_path)
    checks["historical_holdout"] = historical_holdout_diagnostics(weekly)
    checks["eligible_players"] = eligible
    checks["prediction_count"] = len(top3)
    checks["prediction_mean"] = round(float(top3["prediction"].mean()), 6)
    checks["prediction_std"] = round(float(top3["prediction"].std(ddof=0)), 6)
    checks["prediction_min"] = round(float(top3["prediction"].min()), 6)
    checks["prediction_max"] = round(float(top3["prediction"].max()), 6)

    if checks.get("golden_match") is False:
        raise AssertionError(
            "Generated predictions do not match the golden GW dataset: "
            + json.dumps(checks, ensure_ascii=False)
        )

    return PipelineResult(gameweek, eligible, top3, checks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameweek", type=int, required=True)
    parser.add_argument("--golden", type=Path)
    parser.add_argument("--output", type=Path, default=Path("predictions.generated.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.gameweek <= 38:
        raise SystemExit("--gameweek must be between 1 and 38")

    result = run(args.gameweek, args.golden)
    output = build_predictions_json(result.predictions, result.gameweek)

    if not args.dry_run:
        args.output.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps({
        "status": "ok",
        "gameweek": result.gameweek,
        "eligible_players": result.eligible_players,
        "diagnostics": result.diagnostics,
        "output_written": not args.dry_run,
    }, indent=2))


if __name__ == "__main__":
    main()
