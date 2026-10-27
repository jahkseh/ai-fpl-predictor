#!/usr/bin/env python3
"""Maintain a compact history of published FPL model predictions by gameweek."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _snapshot_player(player: dict) -> dict:
    """Retain the fields needed for later, name-matched accuracy evaluation."""
    return {
        "rank": int(player.get("rank", 0)),
        "name": str(player["name"]),
        "prediction": float(player["prediction"]),
        "prediction_p10": player.get("prediction_p10"),
        "prediction_p90": player.get("prediction_p90"),
        "confidence": player.get("confidence"),
        "expected_minutes": player.get("expected_minutes"),
    }


def build_snapshot(predictions: dict) -> dict:
    """Freeze a full player pool when available, with legacy top-three fallback."""
    all_players = predictions.get("all_players")
    if isinstance(all_players, list) and all_players:
        grouped: dict[tuple[str, str], dict] = {}
        for player in all_players:
            home = str(player.get("fixture_home", "")).strip()
            away = str(player.get("fixture_away", "")).strip()
            if not home or not away:
                raise ValueError(
                    f"Full-pool prediction is missing fixture labels: {player.get('name', 'Unknown')}"
                )
            key = (home, away)
            fixture = grouped.setdefault(
                key, {"home": home, "away": away, "players": []}
            )
            fixture["players"].append(_snapshot_player(player))

        fixtures = []
        for key in sorted(grouped):
            fixture = grouped[key]
            fixture["players"].sort(key=lambda player: (player["rank"], player["name"]))
            fixtures.append(fixture)

        recorded_count = sum(len(fixture["players"]) for fixture in fixtures)
        if recorded_count != len(all_players):
            raise AssertionError("Some full-pool player predictions were not preserved")
        scope = "full_player_pool"
    else:
        fixtures = [
            {
                "home": fixture["home"],
                "away": fixture["away"],
                "players": [_snapshot_player(player) for player in fixture["players"]],
            }
            for fixture in predictions["predictions"]
        ]
        scope = "fixture_top_three"
        recorded_count = sum(len(fixture["players"]) for fixture in fixtures)

    return {
        "gameweek": int(predictions["gameweek"]),
        "prediction_scope": scope,
        "players_expected": recorded_count,
        "predictions": fixtures,
    }

def update_history(predictions_path: Path, history_path: Path) -> None:
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    snapshot = build_snapshot(predictions)

    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    else:
        history = {"season": "2026/27", "gameweeks": []}

    gameweeks = [
        item for item in history.get("gameweeks", [])
        if int(item.get("gameweek", -1)) != snapshot["gameweek"]
    ]
    gameweeks.append(snapshot)
    gameweeks.sort(key=lambda item: int(item["gameweek"]))

    history["season"] = predictions.get("season", history.get("season", "2026/27"))
    history["gameweeks"] = gameweeks
    history["gameweek_count"] = len(gameweeks)

    history_path.write_text(
        json.dumps(history, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--history", required=True, type=Path)
    args = parser.parse_args()
    update_history(args.predictions, args.history)


if __name__ == "__main__":
    main()
