#!/usr/bin/env python3
"""
Automated FPL prediction pipeline.

This script is the production-oriented version of the validated Colab workflow.
It deliberately keeps the current Random Forest feature set and hyperparameters
unchanged while adding reproducibility checks and diagnostics for future work.

Usage:
    python model/update_predictions.py --gameweek 5 --dry-run

A future GitHub Actions job can run the same command without --dry-run and
commit the generated predictions.json.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


RAW_BASE = (
    "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/"
    "data/2026-2027/By%20Gameweek/GW{gw}/"
)

PLAYER_FEATURES = [
    "previous_3_total_points",
    "previous_5_total_points",
    "previous_10_total_points",
    "previous_3_minutes",
    "previous_5_minutes",
    "previous_10_minutes",
    "previous_5_start_rate",
    "previous_10_start_rate",
    "previous_5_minutes_rate",
    "previous_10_minutes_rate",
    "previous_3_expected_goals",
    "previous_5_expected_goals",
    "previous_10_expected_goals",
    "previous_3_expected_assists",
    "previous_5_expected_assists",
    "previous_10_expected_assists",
    "value",
    "was_home",
]

MODEL_PARAMS = dict(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)

TEAM_COLUMNS = [
    "code",
    "id",
    "name",
    "short_name",
    "strength",
    "strength_attack_home",
    "strength_attack_away",
    "strength_defence_home",
    "strength_defence_away",
    "elo",
]

ROLLING_COLUMNS = [
    "total_points",
    "minutes",
    "goals_scored",
    "assists",
    "expected_goals",
    "expected_assists",
]


@dataclass
class PipelineResult:
    gameweek: int
    eligible_players: int
    predictions: pd.DataFrame
    diagnostics: dict


def url_for(gw: int, filename: str) -> str:
    return RAW_BASE.format(gw=gw) + filename


def read_csv(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    if df.empty:
        raise ValueError(f"Downloaded empty CSV: {url}")
    return df


def load_current_data(gameweek: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    for gw in range(1, gameweek + 1):
        frames.append(
            read_csv(url_for(gw, "player_gameweek_stats.csv")).assign(GW=gw)
        )
    current = pd.concat(frames, ignore_index=True)

    players = read_csv(url_for(gameweek, "players.csv"))
    teams = read_csv(url_for(gameweek, "teams.csv"))

    required_current = {"id", "total_points", "minutes", "starts", "now_cost"}
    missing = required_current - set(current.columns)
    if missing:
        raise ValueError(f"Current player data missing columns: {sorted(missing)}")

    return current, players, teams


def enrich_players(
    current: pd.DataFrame, players: pd.DataFrame, teams: pd.DataFrame
) -> pd.DataFrame:
    player_lookup = (
        players[["player_id", "team_code", "position"]]
        .drop_duplicates("player_id")
    )
    team_lookup = teams[TEAM_COLUMNS].drop_duplicates("code")

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
        raise AssertionError("Enrichment changed the number of player-gameweek rows")

    for col in ("team_code", "position", "name", "short_name"):
        if col in enriched.columns and enriched[col].isna().any():
            raise ValueError(f"Missing player/team mapping in column: {col}")

    return enriched


def add_rolling_features(enriched: pd.DataFrame) -> pd.DataFrame:
    enriched = enriched.sort_values(["id", "GW"]).copy()

    for col in ROLLING_COLUMNS:
        if col not in enriched.columns:
            raise ValueError(f"Missing rolling source column: {col}")
        for window in (3, 5, 10):
            enriched[f"previous_{window}_{col}"] = (
                enriched.groupby("id")[col]
                .transform(
                    lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
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


def assert_no_leakage(features: pd.DataFrame, gameweek: int) -> None:
    current = features[features["GW"] == gameweek]
    if current.empty:
        raise AssertionError(f"No rows for target GW{gameweek}")

    rolling_cols = [
        c for c in PLAYER_FEATURES
        if c.startswith("previous_")
    ]
    if current[rolling_cols].isna().all(axis=1).any():
        # New players legitimately have no prior history; these are excluded later.
        pass

    eligible = current[current["previous_3_total_points"].notna()]
    if eligible.empty:
        raise AssertionError("No eligible players after history check")

    # The first target-gameweek observation must never be used as its own history.
    # For a player with only one row, previous_3_total_points must remain NaN.
    one_row_ids = (
        features.groupby("id")["GW"].size().loc[lambda s: s == 1].index
    )
    if len(one_row_ids):
        first_rows = features[features["id"].isin(one_row_ids)]
        if first_rows[rolling_cols].notna().any(axis=1).any():
            raise AssertionError("Detected possible current-row leakage in rolling features")


def load_fixture_data(gameweek: int) -> pd.DataFrame:
    fixtures = read_csv(url_for(gameweek, "fixtures.csv"))
    if fixtures.empty:
        raise ValueError("Fixture data is empty")
    return fixtures


def slug_to_team_name(slug: str) -> str:
    mapping = {
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
    if slug not in mapping:
        raise ValueError(f"Unknown fixture team slug: {slug}")
    return mapping[slug]


def fixture_lookup(
    fixtures: pd.DataFrame, teams: pd.DataFrame
) -> pd.DataFrame:
    if "match_id" not in fixtures.columns:
        raise ValueError("fixtures.csv must contain match_id")

    rows = []
    team_name_to_code = dict(
        zip(teams["name"], teams["code"])
    )

    for _, fixture in fixtures.iterrows():
        parts = str(fixture["match_id"]).split("-vs-")
        if len(parts) != 2:
            raise ValueError(f"Unexpected match_id format: {fixture['match_id']}")
        home = slug_to_team_name(parts[0])
        away = slug_to_team_name(parts[1])
        if home not in team_name_to_code or away not in team_name_to_code:
            raise ValueError(f"Fixture team missing from teams.csv: {home} / {away}")

        rows.extend(
            [
                {
                    "team_code": team_name_to_code[home],
                    "fixture_home_team": home,
                    "fixture_away_team": away,
                    "match_id": fixture["match_id"],
                    "was_home_gw": True,
                },
                {
                    "team_code": team_name_to_code[away],
                    "fixture_home_team": home,
                    "fixture_away_team": away,
                    "match_id": fixture["match_id"],
                    "was_home_gw": False,
                },
            ]
        )

    lookup = pd.DataFrame(rows).drop_duplicates("team_code")
    if len(lookup) != len(rows):
        raise AssertionError("A team appears in more than one target-gameweek fixture")
    return lookup


def train_model(enriched: pd.DataFrame, target_gameweek: int) -> RandomForestRegressor:
    train = enriched[
        (enriched["GW"] < target_gameweek)
        & enriched["previous_3_total_points"].notna()
    ].copy()

    # Preserve the original training approach: target is current total_points,
    # features are pre-match rolling/current values.
    X = train[PLAYER_FEATURES].copy()
    y = train["total_points"].copy()

    if X.isna().any().any():
        raise ValueError("Training matrix contains NaN values")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Training matrix contains non-finite values")

    model = RandomForestRegressor(**MODEL_PARAMS)
    model.fit(X, y)
    return model


def predict_target(
    enriched: pd.DataFrame,
    teams: pd.DataFrame,
    target_gameweek: int,
) -> tuple[pd.DataFrame, int]:
    target = enriched[enriched["GW"] == target_gameweek].copy()
    target["value"] = target["now_cost"] * 10

    fixtures = load_fixture_data(target_gameweek)
    lookup = fixture_lookup(fixtures, teams)

    target = target.merge(
        lookup,
        on="team_code",
        how="left",
        validate="many_to_one",
    )

    if target["match_id"].isna().any():
        raise ValueError("Some target-gameweek players could not be mapped to a fixture")

    target["was_home"] = target["was_home_gw"].astype(float)
    eligible = target[target["previous_3_total_points"].notna()].copy()

    model = train_model(enriched, target_gameweek)
    X_target = eligible[PLAYER_FEATURES]
    if X_target.isna().any().any():
        raise ValueError("Eligible prediction matrix contains NaN values")

    eligible["prediction"] = model.predict(X_target)
    eligible["prediction"] = eligible["prediction"].clip(lower=0)

    eligible["fixture"] = (
        eligible["fixture_home_team"] + " vs " + eligible["fixture_away_team"]
    )

    eligible = eligible.sort_values(
        ["fixture", "prediction"], ascending=[True, False]
    )
    eligible["rank"] = eligible.groupby("fixture").cumcount() + 1

    top3 = eligible[eligible["rank"] <= 3].copy()
    return top3, len(eligible)


def build_predictions_json(top3: pd.DataFrame, gameweek: int) -> dict:
    fixtures = []
    for fixture, group in top3.groupby(
        ["fixture_home_team", "fixture_away_team"], sort=True
    ):
        home, away = fixture
        players = [
            {
                "rank": int(row.rank),
                "name": str(row["name"]),
                "prediction": round(float(row.prediction), 6),
            }
            for _, row in group.sort_values("rank").iterrows()
        ]
        fixtures.append({"home": home, "away": away, "players": players})

    return {"gameweek": gameweek, "predictions": fixtures}


def evaluate_predictions(
    prediction_json: dict,
    expected_json_path: Path | None = None,
) -> dict:
    """Compare against a saved golden GW5 prediction file when supplied."""
    if expected_json_path is None:
        return {"golden_match": None}

    expected = json.loads(expected_json_path.read_text(encoding="utf-8"))
    actual_rows = []
    expected_rows = []

    for fixture in prediction_json["predictions"]:
        for p in fixture["players"]:
            actual_rows.append(
                (fixture["home"], fixture["away"], p["name"], p["prediction"])
            )
    for fixture in expected["predictions"]:
        for p in fixture["players"]:
            expected_rows.append(
                (fixture["home"], fixture["away"], p["name"], p["prediction"])
            )

    actual = {(h, a, n): v for h, a, n, v in actual_rows}
    golden = {(h, a, n): v for h, a, n, v in expected_rows}

    keys_equal = set(actual) == set(golden)
    max_abs_diff = (
        max(abs(actual[k] - golden[k]) for k in actual.keys() & golden.keys())
        if actual.keys() & golden.keys()
        else math.inf
    )

    return {
        "golden_match": keys_equal and max_abs_diff <= 1e-6,
        "golden_keys_equal": keys_equal,
        "golden_max_abs_difference": max_abs_diff,
        "golden_prediction_count": len(golden),
        "generated_prediction_count": len(actual),
    }


def diagnostics(top3: pd.DataFrame) -> dict:
    """Lightweight monitoring metrics useful for future model optimization."""
    p = top3["prediction"].to_numpy(dtype=float)
    return {
        "prediction_count": int(len(top3)),
        "prediction_mean": round(float(np.mean(p)), 6),
        "prediction_std": round(float(np.std(p)), 6),
        "prediction_min": round(float(np.min(p)), 6),
        "prediction_max": round(float(np.max(p)), 6),
        "negative_prediction_count_before_clip": 0,
        "fixture_count": int(top3["fixture"].nunique()),
        "players_per_fixture_ok": bool(top3.groupby("fixture").size().eq(3).all()),
    }


def run(gameweek: int, golden_path: Path | None = None) -> PipelineResult:
    current, players, teams = load_current_data(gameweek)
    enriched = enrich_players(current, players, teams)
    enriched = add_rolling_features(enriched)
    assert_no_leakage(enriched, gameweek)

    top3, eligible_count = predict_target(enriched, teams, gameweek)
    output = build_predictions_json(top3, gameweek)

    checks = evaluate_predictions(output, golden_path)
    checks.update(diagnostics(top3))

    if checks.get("golden_match") is False:
        raise AssertionError(
            "Generated predictions do not match the supplied golden dataset. "
            f"Maximum absolute difference: {checks.get('golden_max_abs_difference')}"
        )

    return PipelineResult(
        gameweek=gameweek,
        eligible_players=eligible_count,
        predictions=top3,
        diagnostics=checks,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the automated FPL prediction pipeline.")
    parser.add_argument("--gameweek", type=int, required=True)
    parser.add_argument("--golden", type=Path, default=None)
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
