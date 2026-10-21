import unittest

from measure_prediction_accuracy import calculate_metrics, snapshot_players
from publish_prediction_history import build_snapshot


class PredictionHistoryTests(unittest.TestCase):
    def test_snapshot_keeps_top_three_and_full_player_pool(self):
        data = {
            "gameweek": 9,
            "predictions": [{"home": "A", "away": "B", "players": [
                {"rank": 1, "name": "Player A", "prediction": 5.0},
                {"rank": 2, "name": "Player B", "prediction": 4.0},
                {"rank": 3, "name": "Player C", "prediction": 3.0},
            ]}],
            "all_players": [
                {"rank": 1, "name": "Player A", "prediction": 5.0, "prediction_p10": 1.0, "prediction_p90": 9.0, "expected_minutes": 80, "position": "Forward", "team": "A"},
                {"rank": 2, "name": "Player D", "prediction": 2.0, "prediction_p10": 0.0, "prediction_p90": 4.0, "expected_minutes": 30, "position": "Midfielder", "team": "B"},
            ],
        }
        snapshot = build_snapshot(data)
        self.assertEqual(len(snapshot["predictions"][0]["players"]), 3)
        self.assertEqual(len(snapshot["all_players"]), 2)
        self.assertEqual(snapshot["all_players"][1]["name"], "Player D")
        self.assertEqual(snapshot["all_players"][1]["expected_minutes"], 30)

    def test_legacy_history_is_identified_as_top_three_only(self):
        players, scope = snapshot_players({"predictions": [{"players": [{"name": "Player A", "prediction": 5.0}]}]})
        self.assertEqual(scope, "top_three_per_fixture_legacy")
        self.assertEqual(len(players), 1)

    def test_full_pool_history_is_preferred(self):
        players, scope = snapshot_players({
            "all_players": [{"name": "Player A", "prediction": 1}],
            "predictions": [{"players": [{"name": "Player B", "prediction": 2}]}],
        })
        self.assertEqual(scope, "full_player_pool")
        self.assertEqual(players[0]["name"], "Player A")

    def test_metrics_report_scope_and_error(self):
        result = calculate_metrics([
            {"prediction": 5.0, "actual": 4.0, "p10": 2.0, "p90": 6.0},
            {"prediction": 2.0, "actual": 3.0, "p10": 1.0, "p90": 4.0},
        ], 9, "full_player_pool")
        self.assertEqual(result["mae"], 1.0)
        self.assertEqual(result["players_evaluated"], 2)
        self.assertEqual(result["coverage_scope"], "full_player_pool")
        self.assertEqual(result["p10_p90_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
