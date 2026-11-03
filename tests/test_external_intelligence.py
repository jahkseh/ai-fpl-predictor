import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from model.external_intelligence import (
    MAX_ODDS_REQUESTS_PER_MONTH,
    collect_cup_minutes,
    collect_odds,
)


class ExternalIntelligenceTests(unittest.TestCase):
    def test_cup_collector_uses_recent_cup_matches_and_player_minutes(self):
        now = datetime(2026, 11, 3, 12, tzinfo=timezone.utc)

        def fake_get(path, api_key, params=None):
            if path == "/leagues":
                return {"data": {"leagues": [
                    {"id": "l_cup", "name": "EFL Cup", "seasons": ["2026/2027"]},
                    {"id": "l_pl", "name": "Premier League", "seasons": ["2026/2027"]},
                ]}}
            if path == "/leagues/l_cup/matches":
                return {"data": {"matches": [{
                    "id": "m_1", "date": "2026-11-02", "status": "finished",
                    "home_team": {"name": "Liverpool"}, "away_team": {"name": "Chelsea"},
                }]}}
            if path == "/matches/m_1/advanced/players":
                return {"data": {"players": [{
                    "player": {"id": "p_1", "name": "M. Salah"},
                    "team_id": "t_1", "minutes_played": 67,
                }]}}
            raise AssertionError(path)

        with patch("model.external_intelligence.pitch_get", side_effect=fake_get):
            result = collect_cup_minutes("test", now=now)

        self.assertEqual(result["matches_checked"], 1)
        self.assertEqual(result["appearances"][0]["minutes"], 67)
        self.assertEqual(result["appearances"][0]["provider_player_id"], "p_1")

    def test_odds_collector_stops_at_monthly_cap_without_requests(self):
        now = datetime(2026, 11, 3, 12, tzinfo=timezone.utc)
        usage = {"month": "2026-11", "month_count": MAX_ODDS_REQUESTS_PER_MONTH}
        with patch("model.external_intelligence.fetch_json") as fetch:
            fetch.return_value = {"data": {"events": [{"event_id": "evt_1"}]}}
            snapshot, updated_usage = collect_odds("test", usage=usage, now=now)

        self.assertEqual(snapshot["snapshots"], [])
        self.assertEqual(updated_usage["month_count"], MAX_ODDS_REQUESTS_PER_MONTH)
        self.assertEqual(fetch.call_count, 1)  # Only the unmetered events catalogue.

    def test_failed_odds_attempt_still_consumes_local_safety_budget(self):
        now = datetime(2026, 11, 3, 12, tzinfo=timezone.utc)
        usage = {"month": "2026-11", "month_count": 0}
        with patch("model.external_intelligence.fetch_json") as fetch:
            fetch.side_effect = [
                {"data": {"events": [{"event_id": "evt_1"}]}},
                RuntimeError("temporary provider failure"),
            ]
            snapshot, updated_usage = collect_odds("test", usage=usage, now=now)

        self.assertEqual(snapshot["snapshots"], [])
        self.assertEqual(updated_usage["month_count"], 1)


if __name__ == "__main__":
    unittest.main()
