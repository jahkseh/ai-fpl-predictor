#!/usr/bin/env python3
"""Build a timestamped, source-attributed ScoutXI intelligence snapshot.

Phase 1 intentionally uses the public FPL API only. It does not invent injury
news, cup appearances, odds, xG, or player-level projections. Those sources are
added only after their availability, licensing and schemas have been validated.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

BASE = "https://fantasy.premierleague.com/api/"
USER_AGENT = "ScoutXI-Intelligence/1.0 (+https://jahkseh.github.io/ai-fpl-predictor/)"


def get_json(path: str):
    request = Request(BASE + path, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(request, timeout=25) as response:
        if response.status != 200:
            raise RuntimeError(f"FPL API returned HTTP {response.status} for {path}")
        return json.load(response)


def build_snapshot() -> dict:
    collected_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    bootstrap = get_json("bootstrap-static/")
    fixtures_raw = get_json("fixtures/")

    teams = {int(team["id"]): team for team in bootstrap.get("teams", [])}
    events = bootstrap.get("events", [])
    next_event = next((event for event in events if event.get("is_next")), None)
    if next_event is None:
        next_event = next((event for event in events if not event.get("finished")), None)

    players = []
    for player in bootstrap.get("elements", []):
        team = teams.get(int(player.get("team") or 0), {})
        players.append({
            "id": player.get("id"),
            "name": " ".join(part for part in [player.get("first_name"), player.get("second_name")] if part),
            "team": team.get("name"),
            "team_short_name": team.get("short_name"),
            "position_id": player.get("element_type"),
            "price": (player.get("now_cost") / 10) if isinstance(player.get("now_cost"), (int, float)) else None,
            "ownership_percent": _number(player.get("selected_by_percent")),
            "form": _number(player.get("form")),
            "points_per_game": _number(player.get("points_per_game")),
            "expected_goals": _number(player.get("expected_goals")),
            "expected_assists": _number(player.get("expected_assists")),
            "expected_goal_involvements": _number(player.get("expected_goal_involvements")),
            "minutes": _number(player.get("minutes")),
            "status": player.get("status"),
            "news": player.get("news") or "",
            "chance_of_playing_next_round": player.get("chance_of_playing_next_round"),
            "chance_of_playing_this_round": player.get("chance_of_playing_this_round"),
            "source": "FPL bootstrap-static",
        })

    fixtures = []
    for fixture in fixtures_raw:
        home = teams.get(int(fixture.get("team_h") or 0), {})
        away = teams.get(int(fixture.get("team_a") or 0), {})
        fixtures.append({
            "id": fixture.get("id"),
            "gameweek": fixture.get("event"),
            "kickoff_time": fixture.get("kickoff_time"),
            "finished": bool(fixture.get("finished")),
            "started": bool(fixture.get("started")),
            "home_team": home.get("name"),
            "away_team": away.get("name"),
            "home_team_id": fixture.get("team_h"),
            "away_team_id": fixture.get("team_a"),
            "source": "FPL fixtures endpoint",
        })

    return {
        "schema_version": 1,
        "collected_at": collected_at,
        "season": "2026/27",
        "target_gameweek": next_event.get("id") if next_event else None,
        "sources": [
            {"name": "Official Fantasy Premier League API", "endpoint": "bootstrap-static/", "status": "ok"},
            {"name": "Official Fantasy Premier League API", "endpoint": "fixtures/", "status": "ok"},
            {"name": "Official Fantasy Premier League API", "endpoint": "bootstrap-static/ player expected_goals, expected_assists and expected_goal_involvements fields", "status": "ok"},
        ],
        "coverage": {
            "players": len(players),
            "fixtures": len(fixtures),
            "injury_status_fields": True,
            "confirmed_team_news": False,
            "cup_appearances": False,
            "expected_goals_data": any(p.get("expected_goals") is not None or p.get("expected_assists") is not None for p in players),
            "expected_goals_source": "Official FPL bootstrap fields; season aggregates, not independent match-level xG/xA",
            "betting_market_data": False,
            "reason": "Includes official FPL season xG/xA fields where published. Independent confirmed team-news sources, cup appearances and betting market data are not integrated; these remain explicitly unavailable.",
        },
        "players": players,
        "fixtures": fixtures,
    }


def _number(value):
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def main() -> None:
    output = Path(os.environ.get("SCOUTXI_INTELLIGENCE_OUTPUT", "intelligence_snapshot.json"))
    snapshot = build_snapshot()
    if not snapshot["players"]:
        raise SystemExit("Refusing to publish an empty player snapshot")
    if not snapshot["sources"] or any(source["status"] != "ok" for source in snapshot["sources"]):
        raise SystemExit("Source validation failed")
    output.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output}: {len(snapshot['players'])} players, {len(snapshot['fixtures'])} fixtures, collected {snapshot['collected_at']}")


if __name__ == "__main__":
    main()
