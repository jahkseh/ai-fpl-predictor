#!/usr/bin/env python3
"""Publish the current FPL gameweek fixture list and live/finished scores."""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

BASE = "https://fantasy.premierleague.com/api/"
def fetch(path):
    request = Request(BASE + path, headers={"User-Agent": "ScoutXI/1.0 (public FPL fixtures)"})
    with urlopen(request, timeout=25) as response:
        return json.loads(response.read().decode("utf-8"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="current_fixtures.json")
    args = parser.parse_args()
    bootstrap = fetch("bootstrap-static/")
    fixtures = fetch("fixtures/")
    events = bootstrap.get("events", [])
    current = next((e for e in events if e.get("is_current")), None)
    if current is None:
        active_ids = [int(f["event"]) for f in fixtures if f.get("event") and f.get("started") and not f.get("finished")]
        if active_ids:
            current = next((e for e in events if int(e["id"]) == max(active_ids)), None)
    if current is None:
        finished_ids = [int(e["id"]) for e in events if e.get("finished")]
        current = next((e for e in events if int(e["id"]) == max(finished_ids)), None) if finished_ids else None
    if current is None:
        raise RuntimeError("FPL API did not identify a current or most recently completed gameweek")
    gw = int(current["id"])
    teams = {int(t["id"]): t for t in bootstrap.get("teams", [])}
    rows = []
    for fixture in fixtures:
        if int(fixture.get("event") or 0) != gw:
            continue
        home, away = teams.get(int(fixture["team_h"])), teams.get(int(fixture["team_a"]))
        if not home or not away:
            continue
        rows.append({
            "id": int(fixture["id"]), "gameweek": gw,
            "home": home["name"], "away": away["name"],
            "home_short": home.get("short_name", ""), "away_short": away.get("short_name", ""),
            "home_logo": f"https://resources.premierleague.com/premierleague/badges/50/t{home['id']}.png",
            "away_logo": f"https://resources.premierleague.com/premierleague/badges/50/t{away['id']}.png",
            "kickoff": fixture.get("kickoff_time"),
            "started": bool(fixture.get("started")), "finished": bool(fixture.get("finished")),
            "finished_provisional": bool(fixture.get("finished_provisional")),
            "minutes": fixture.get("minutes"),
            "home_score": fixture.get("team_h_score"), "away_score": fixture.get("team_a_score"),
        })
    rows.sort(key=lambda x: (x["kickoff"] or "", x["id"]))
    if not rows:
        raise RuntimeError(f"No official fixtures returned for current GW{gw}; refusing to publish an empty schedule")
    payload = {
        "season": "2026/27", "gameweek": gw,
        "event_name": current.get("name", f"Gameweek {gw}"),
        "deadline": current.get("deadline_time"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixtures": rows,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Published {len(rows)} official fixtures for GW{gw}")

if __name__ == "__main__":
    main()
