#!/usr/bin/env python3
"""Collect optional, source-attributed ScoutXI external intelligence.

This collector is deliberately fail-closed. Third-party feeds are not requested
or published unless their GitHub Actions secret exists AND a repository
maintainer has explicitly enabled the matching public-use approval variable.
It never changes the prediction model.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

PITCH_BASE = "https://api.pitchapi.dev/v1"
ODDS_BASE = "https://api.ukoddsapi.com/v1/football"
DEFAULT_OUTPUT = "external_intelligence_snapshot.json"
MAX_ODDS_REQUESTS_PER_MONTH = 240  # Leave a safety margin below the 300 free-tier quota.
CUP_NAME = re.compile(
    r"(\bfa cup\b|\b(e?fl|carabao) cup\b|\bchampions league\b|"
    r"\beuropa league\b|\bconference league\b|\buefa super cup\b)",
    re.IGNORECASE,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def fetch_json(url: str, *, headers: dict[str, str], timeout: int = 25) -> dict:
    request = Request(url, headers={"Accept": "application/json", **headers})
    with urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"Provider returned HTTP {response.status}")
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise RuntimeError("Provider response was not a JSON object")
    if payload.get("error"):
        raise RuntimeError("Provider returned an error response")
    return payload


def pitch_get(path: str, api_key: str, params: dict | None = None) -> dict:
    query = f"?{urlencode(params)}" if params else ""
    return fetch_json(
        f"{PITCH_BASE}{path}{query}",
        headers={"X-API-KEY": api_key},
    )


def _data(payload: dict):
    if "data" not in payload:
        raise RuntimeError("Provider response is missing its data envelope")
    return payload["data"]


def collect_cup_minutes(api_key: str, *, now: datetime | None = None) -> dict:
    """Collect recent player minutes from listed domestic/UEFA cup competitions."""
    now = now or utc_now()
    season = f"{now.year}/{now.year + 1}" if now.month >= 7 else f"{now.year - 1}/{now.year}"
    league_data = _data(pitch_get("/leagues", api_key))
    leagues = league_data.get("leagues", []) if isinstance(league_data, dict) else []
    cup_leagues = [
        league for league in leagues
        if isinstance(league, dict)
        and CUP_NAME.search(str(league.get("name", "")))
        and season in (league.get("seasons") or [])
        and league.get("id")
    ]

    cutoff = now - timedelta(days=14)
    matches: list[dict] = []
    for league in cup_leagues:
        league_id = str(league["id"])
        match_data = _data(pitch_get(f"/leagues/{league_id}/matches", api_key, {"season": season}))
        league_matches = match_data.get("matches", []) if isinstance(match_data, dict) else []
        for match in league_matches:
            if not isinstance(match, dict) or not match.get("id"):
                continue
            raw_date = match.get("time_utc") or match.get("date")
            if not raw_date:
                continue
            try:
                match_time = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
                if match_time.tzinfo is None:
                    match_time = match_time.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if match_time < cutoff or match_time > now or match.get("status") not in {"finished", "completed"}:
                continue
            matches.append({**match, "competition": league.get("name"), "competition_id": league_id})

    appearances = []
    for match in matches:
        match_id = str(match["id"])
        try:
            analytics = _data(pitch_get(f"/matches/{match_id}/advanced/players", api_key))
        except (HTTPError, URLError, RuntimeError):
            # A match without advanced analytics is missing coverage, not zero minutes.
            continue
        players = analytics.get("players", []) if isinstance(analytics, dict) else []
        for row in players:
            if not isinstance(row, dict) or not isinstance(row.get("player"), dict):
                continue
            minutes = row.get("minutes_played")
            if not isinstance(minutes, (int, float)) or minutes < 0:
                continue
            appearances.append({
                "provider_player_id": row["player"].get("id"),
                "player_name": row["player"].get("name"),
                "team_id": row.get("team_id"),
                "minutes": int(minutes),
                "match_id": match_id,
                "match_date": match.get("time_utc") or match.get("date"),
                "competition": match.get("competition"),
                "home_team": (match.get("home_team") or {}).get("name"),
                "away_team": (match.get("away_team") or {}).get("name"),
                "source": "PitchAPI",
            })

    return {
        "status": "ok",
        "provider": "PitchAPI",
        "season": season,
        "lookback_days": 14,
        "competitions_checked": len(cup_leagues),
        "matches_checked": len(matches),
        "appearances": appearances,
        "collected_at": now.isoformat().replace("+00:00", "Z"),
    }


def _month_key(now: datetime) -> str:
    return now.strftime("%Y-%m")


def collect_odds(api_key: str, *, usage: dict, now: datetime | None = None) -> tuple[dict, dict]:
    """Take a quota-capped snapshot of upcoming PL fixture core odds.

    The caller must enable public use only after confirming the provider's
    display/derived-data licence for ScoutXI. We deliberately request core
    fixture markets only and never infer an individual injury from odds.
    """
    now = now or utc_now()
    month = _month_key(now)
    count = usage.get("month_count", 0) if usage.get("month") == month else 0
    events_payload = fetch_json(
        f"{ODDS_BASE}/events?{urlencode({'from': now.date().isoformat(), 'to': (now + timedelta(days=2)).date().isoformat(), 'league': 'premier league'})}",
        headers={"X-Api-Key": api_key},
    )
    events_data = events_payload.get("data", events_payload)
    events = events_data.get("events", events_data if isinstance(events_data, list) else [])
    events = [event for event in events if isinstance(event, dict) and event.get("event_id")]
    events.sort(key=lambda item: str(item.get("start_time") or item.get("commence_time") or item.get("date") or ""))

    snapshots = []
    for event in events:
        if count >= MAX_ODDS_REQUESTS_PER_MONTH:
            break
        event_id = str(event["event_id"])
        # Count every attempted odds request: failed HTTP responses may still
        # consume provider quota, so the guard must be conservative.
        count += 1
        try:
            payload = fetch_json(
                f"{ODDS_BASE}/events/{event_id}/odds?package=core&odds_format=decimal",
                headers={"X-Api-Key": api_key},
            )
        except (HTTPError, URLError, RuntimeError):
            continue
        snapshots.append({
            "event_id": event_id,
            "home_team": event.get("home_team"),
            "away_team": event.get("away_team"),
            "start_time": event.get("start_time") or event.get("commence_time") or event.get("date"),
            "captured_at": now.isoformat().replace("+00:00", "Z"),
            "odds": payload.get("data", payload),
        })

    return ({
        "status": "ok",
        "provider": "UK Odds API",
        "market_scope": "core fixture markets",
        "events_checked": len(events),
        "snapshots": snapshots,
        "quota_limit": MAX_ODDS_REQUESTS_PER_MONTH,
        "collected_at": now.isoformat().replace("+00:00", "Z"),
    }, {"month": month, "month_count": count})


def build_snapshot() -> dict:
    now = utc_now()
    snapshot = {
        "schema_version": 1,
        "collected_at": now.isoformat().replace("+00:00", "Z"),
        "prediction_model_changed": False,
        "sources": {},
        "coverage": {
            "cup_minutes": False,
            "betting_odds": False,
            "independent_press_conference_news": False,
            "injury_reports": "official FPL availability/news fields only",
        },
        "cup_workload": None,
        "odds_monitor": None,
        "notes": [
            "External signals are informational only and do not alter ScoutXI predictions.",
            "Missing provider data is null/unavailable, never zero.",
            "Odds movement is a team/match-level investigation signal, not proof of an individual injury.",
        ],
    }

    pitch_key = os.environ.get("PITCHAPI_API_KEY", "").strip()
    pitch_approved = os.environ.get("PITCHAPI_PUBLIC_USE_APPROVED", "").lower() == "true"
    if pitch_key and pitch_approved:
        try:
            snapshot["cup_workload"] = collect_cup_minutes(pitch_key, now=now)
            snapshot["coverage"]["cup_minutes"] = True
            snapshot["sources"]["pitchapi"] = {"status": "ok", "public_use_approved": True}
        except (HTTPError, URLError, RuntimeError, ValueError) as exc:
            snapshot["sources"]["pitchapi"] = {"status": "unavailable", "error_type": type(exc).__name__}
    else:
        snapshot["sources"]["pitchapi"] = {
            "status": "disabled",
            "reason": "Missing secret or explicit public-use approval variable",
        }

    odds_key = os.environ.get("UK_ODDS_API_KEY", "").strip()
    odds_approved = os.environ.get("UK_ODDS_PUBLIC_USE_APPROVED", "").lower() == "true"
    usage_path = Path(os.environ.get("SCOUTXI_ODDS_USAGE_FILE", "odds_usage.json"))
    usage = json.loads(usage_path.read_text(encoding="utf-8")) if usage_path.exists() else {}
    if odds_key and odds_approved:
        try:
            odds, usage = collect_odds(odds_key, usage=usage, now=now)
            snapshot["odds_monitor"] = odds
            snapshot["coverage"]["betting_odds"] = True
            snapshot["sources"]["uk_odds_api"] = {"status": "ok", "public_use_approved": True}
        except (HTTPError, URLError, RuntimeError, ValueError) as exc:
            snapshot["sources"]["uk_odds_api"] = {"status": "unavailable", "error_type": type(exc).__name__}
    else:
        snapshot["sources"]["uk_odds_api"] = {
            "status": "disabled",
            "reason": "Missing secret or explicit public-use approval variable",
        }

    usage_path.write_text(json.dumps(usage, indent=2) + "\n", encoding="utf-8")
    return snapshot


def main() -> None:
    output = Path(os.environ.get("SCOUTXI_EXTERNAL_INTELLIGENCE_OUTPUT", DEFAULT_OUTPUT))
    snapshot = build_snapshot()
    output.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        "External intelligence snapshot written; "
        f"cup_minutes={snapshot['coverage']['cup_minutes']}, "
        f"odds={snapshot['coverage']['betting_odds']}, "
        "prediction_model_changed=false"
    )


if __name__ == "__main__":
    main()
