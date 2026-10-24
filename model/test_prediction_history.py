"""Regression tests for frozen prediction history coverage and compatibility."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from model.publish_prediction_history import build_snapshot, update_history


class PredictionHistoryTests(unittest.TestCase):
    def full_pool_predictions(self, gameweek: int = 8, first_prediction: float = 5.2) -> dict:
        return {
            "gameweek": gameweek,
            "season": "2026/27",
            "predictions": [
                {
                    "home": "Arsenal",
                    "away": "Chelsea",
                    "players": [
                        {
                            "rank": 1,
                            "name": "Player One",
                            "prediction": first_prediction,
                            "prediction_p10": 2.1,
                            "prediction_p90": 8.4,
                            "confidence": "medium",
                            "expected_minutes": 82,
                        }
                    ],
                }
            ],
            "all_players": [
                {
                    "rank": 1,
                    "name": "Player One",
                    "prediction": first_prediction,
                    "prediction_p10": 2.1,
                    "prediction_p90": 8.4,
                    "confidence": "medium",
                    "expected_minutes": 82,
                    "fixture_home": "Arsenal",
                    "fixture_away": "Chelsea",
                },
                {
                    "rank": 2,
                    "name": "Player Two",
                    "prediction": 4.0,
                    "prediction_p10": 1.0,
                    "prediction_p90": 6.7,
                    "confidence": "high",
                    "expected_minutes": 90,
                    "fixture_home": "Arsenal",
                    "fixture_away": "Chelsea",
                },
                {
                    "rank": 1,
                    "name": "Player Three",
                    "prediction": 3.1,
                    "prediction_p10": 0.8,
                    "prediction_p90": 5.8,
                    "confidence": "low",
                    "expected_minutes": 54,
                    "fixture_home": "Liverpool",
                    "fixture_away": "Everton",
                },
            ],
        }

    def test_full_player_pool_is_frozen_across_fixtures(self) -> None:
        snapshot = build_snapshot(self.full_pool_predictions())
        self.assertEqual(snapshot["prediction_scope"], "full_player_pool")
        self.assertEqual(snapshot["players_expected"], 3)
        self.assertEqual(
            {(fixture["home"], fixture["away"]) for fixture in snapshot["predictions"]},
            {("Arsenal", "Chelsea"), ("Liverpool", "Everton")},
        )
        self.assertEqual(
            sum(len(fixture["players"]) for fixture in snapshot["predictions"]), 3
        )
        first_fixture = next(
            fixture for fixture in snapshot["predictions"] if fixture["home"] == "Arsenal"
        )
        self.assertEqual(first_fixture["players"][0]["name"], "Player One")
        self.assertEqual(first_fixture["players"][0]["expected_minutes"], 82)

    def test_legacy_snapshot_without_all_players_stays_compatible(self) -> None:
        legacy = {
            "gameweek": 7,
            "predictions": [
                {
                    "home": "Arsenal",
                    "away": "Chelsea",
                    "players": [
                        {
                            "rank": 1,
                            "name": "Player One",
                            "prediction": 5.2,
                            "prediction_p10": 2.1,
                            "prediction_p90": 8.4,
                            "confidence": "medium",
                            "expected_minutes": 82,
                        }
                    ],
                }
            ],
        }
        snapshot = build_snapshot(legacy)
        self.assertEqual(snapshot["prediction_scope"], "fixture_top_three")
        self.assertEqual(snapshot["players_expected"], 1)
        self.assertEqual(snapshot["predictions"][0]["players"][0]["name"], "Player One")

    def test_history_preserves_gameweeks_and_replaces_same_gameweek(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "predictions.json"
            history = root / "prediction_history.json"

            source.write_text(
                json.dumps(self.full_pool_predictions(gameweek=7)),
                encoding="utf-8",
            )
            update_history(source, history)
            source.write_text(
                json.dumps(self.full_pool_predictions(gameweek=8)),
                encoding="utf-8",
            )
            update_history(source, history)
            source.write_text(
                json.dumps(self.full_pool_predictions(gameweek=8, first_prediction=6.4)),
                encoding="utf-8",
            )
            update_history(source, history)

            saved = json.loads(history.read_text(encoding="utf-8"))
            self.assertEqual([item["gameweek"] for item in saved["gameweeks"]], [7, 8])
            self.assertEqual(saved["gameweek_count"], 2)
            self.assertEqual(saved["gameweeks"][1]["prediction_scope"], "full_player_pool")
            first_fixture = next(
                fixture
                for fixture in saved["gameweeks"][1]["predictions"]
                if fixture["home"] == "Arsenal"
            )
            self.assertEqual(first_fixture["players"][0]["prediction"], 6.4)

    def test_full_pool_requires_fixture_labels(self) -> None:
        predictions = self.full_pool_predictions()
        predictions["all_players"][0].pop("fixture_home")
        with self.assertRaisesRegex(ValueError, "missing fixture labels"):
            build_snapshot(predictions)


if __name__ == "__main__":
    unittest.main()
