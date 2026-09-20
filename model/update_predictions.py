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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


HISTORICAL_URL = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
    "master/data/2025-26/gws/merged_gw.csv"
)
CURRENT_BASE = (
    "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/"
    "data/2026-2027/By%20Gameweek/GW{gw}/"
)

PLAYER_FEATURES = [
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

MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 5,
    "random_state": 42,
    "n_jobs": -1,
}

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
    first_columns = [c for c in ["name", "value", "was_home"] if c in df.columns]

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
        frames.append(
            read_csv(current_url(gw, "player_gameweek_stats.csv")).assign(GW=gw)
        )
    current = pd.concat(frames, ignore_index=True)

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
    previous = [c for c in PLAYER_FEATURES if c.startswith("previous_")]
    if first_rows[previous].notna().any(axis=1).any():
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

        rows.extend([
            {
                "team_code": team_name_to_code[home],
                "fixture_home_team": home,
                "fixture_away_team": away,
                "match_id": fixture["match_id"],
                "was_home_gw": 1,
            },
            {
                "team_code": team_name_to_code[away],
                "fixture_home_team": home,
                "fixture_away_team": away,
                "match_id": fixture["match_id"],
                "was_home_gw": 0,
            },
        ])

    lookup = pd.DataFrame(rows)
    if lookup["team_code"].duplicated().any():
        raise AssertionError("A team appears in more than one target fixture")
    if len(lookup) != 20:
        raise AssertionError(f"Expected 20 team fixture mappings, got {len(lookup)}")
    return lookup


def train_validated_model(weekly: pd.DataFrame) -> RandomForestRegressor:
    train = weekly[
        weekly["GW"].between(2, 32)
        & weekly["previous_3_total_points"].notna()
    ].copy()

    X = train[PLAYER_FEATURES]
    y = train["total_points"]

    if X.isna().any().any():
        raise ValueError("Historical training features contain missing values")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Historical training features contain non-finite values")

    model = RandomForestRegressor(**MODEL_PARAMS)
    model.fit(X, y)
    return model


def historical_holdout_diagnostics(weekly: pd.DataFrame) -> dict:
    train = weekly[
        weekly["GW"].between(2, 32)
        & weekly["previous_3_total_points"].notna()
    ]
    test = weekly[weekly["GW"].between(33, 38)].copy()

    X_train = train[PLAYER_FEATURES]
    y_train = train["total_points"]
    X_test = test[PLAYER_FEATURES]
    y_test = test["total_points"]

    if X_test.isna().any().any():
        raise ValueError("Historical holdout contains missing features")

    model = RandomForestRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "mae": round(float(mean_absolute_error(y_test, pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 4),
        "r2": round(float(r2_score(y_test, pred)), 4),
        "expected_mae": 1.0131,
        "expected_rmse": 1.9953,
        "expected_r2": 0.3152,
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

    # Keep the exact GW5 rule: players without complete pre-gameweek history
    # are retained in the source data but excluded from model prediction.
    eligible = target.dropna(subset=["previous_3_total_points"]).copy()

    if eligible[PLAYER_FEATURES].isna().any().any():
        raise ValueError("Eligible target matrix contains missing values")

    model = train_validated_model(weekly_historical)
    eligible["prediction"] = np.clip(model.predict(eligible[PLAYER_FEATURES]), 0, None)
    eligible["fixture"] = (
        eligible["fixture_home_team"] + " vs " + eligible["fixture_away_team"]
    )

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
                "name": str(row["web_name"]),
                "prediction": round(float(row["prediction"]), 6),
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
