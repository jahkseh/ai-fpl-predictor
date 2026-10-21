#!/usr/bin/env python3
"""Maintain a compact history of published FPL model predictions by gameweek."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_snapshot(predictions: dict) -> dict:
    # Keep the existing fixture/top-three format for the website, and preserve
    # the full player pool separately for honest whole-pool evaluation.
    snapshot = {
        "gameweek": int(predictions["gameweek"]),
        "predictions": [
            {
                "home": fixture["home"],
                "away": fixture["away"],
                "players": [
                    {
                        "rank": int(player["rank"]),
                        "name": str(player["name"]),
                        "prediction": float(player["prediction"]),
                        "prediction_p10": player.get("prediction_p10"),
                        "prediction_p90": player.get("prediction_p90"),
                        "confidence": player.get("confidence"),
                    }
                    for player in fixture["players"]
                ],
            }
            for fixture in predictions["predictions"]
        ],
    }

    all_players = predictions.get("all_players")
    if isinstance(all_players, list):
        snapshot["all_players"] = [
            {
                "rank": int(player["rank"]) if player.get("rank") is not None else None,
                "name": str(player.get("name", "")).strip(),
                "prediction": float(player["prediction"]),
                "prediction_p10": player.get("prediction_p10"),
                "prediction_p90": player.get("prediction_p90"),
                "confidence": player.get("confidence"),
                "expected_minutes": player.get("expected_minutes"),
                "position": player.get("position"),
                "team": player.get("team"),
                "fixture_home": player.get("fixture_home"),
                "fixture_away": player.get("fixture_away"),
                "price": player.get("price"),
                "ownership": player.get("ownership"),
            }
            for player in all_players
            if str(player.get("name", "")).strip() and player.get("prediction") is not None
        ]

    return snapshot


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
