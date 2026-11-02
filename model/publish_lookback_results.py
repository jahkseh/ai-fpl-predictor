#!/usr/bin/env python3
"""Publish browser-ready Look Back comparisons using official FPL results."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API = "https://fantasy.premierleague.com/api"


def fetch_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "ScoutXI-LookBack/1.0"})
    with urlopen(request, timeout=25) as response:
        return json.load(response)


def norm(value: object) -> str:
    value = unicodedata.normalize("NFKD", str(value or ""))
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def team_aliases(value: object) -> set[str]:
    key = norm(value)
    aliases = {key}
    known = {
        "manchesterunited": {"manutd", "manunited"},
        "manchestercity": {"mancity"},
        "tottenhamhotspur": {"spurs", "tottenham"},
        "newcastleunited": {"newcastle"},
        "westhamunited": {"westham"},
        "wolverhamptonwanderers": {"wolves"},
        "brightonandhovealbion": {"brighton"},
        "nottinghamforest": {"nottmforest", "forest"},
        "leicestercity": {"leicester"},
    }
    for canonical, extra in known.items():
        if key == canonical or key in extra:
            aliases.add(canonical)
            aliases.update(extra)
    return aliases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    history = json.loads(args.history.read_text(encoding="utf-8"))
    payload = {
        "season": history.get("season"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": "official_fpl_api",
        "gameweeks": [],
    }
    snapshots = sorted(history.get("gameweeks", []), key=lambda row: int(row.get("gameweek", 0)))
    if not snapshots:
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return

    try:
        bootstrap = fetch_json(f"{API}/bootstrap-static/")
    except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not fetch official FPL bootstrap: {type(exc).__name__}") from exc

    events = {int(row["id"]): row for row in bootstrap.get("events", [])}
    elements = {int(row["id"]): row for row in bootstrap.get("elements", [])}
    teams = {int(row["id"]): row.get("name", "Unknown") for row in bootstrap.get("teams", [])}

    for snapshot in snapshots:
        gw = int(snapshot.get("gameweek", 0))
        event = events.get(gw)
        if not event or not event.get("finished"):
            continue
        try:
            live = fetch_json(f"{API}/event/{gw}/live/")
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError):
            continue

        actual_by_alias: dict[str, list[dict]] = {}
        for item in live.get("elements", []):
            player = elements.get(int(item.get("id", -1)))
            stats = item.get("stats", {})
            if not player or stats.get("total_points") is None:
                continue
            actual = {
                "id": int(player["id"]),
                "name": player.get("web_name") or f"{player.get('first_name', '')} {player.get('second_name', '')}".strip(),
                "team": teams.get(int(player.get("team", 0)), "Unknown"),
                "minutes": int(stats.get("minutes") or 0),
                "actual": float(stats["total_points"]),
            }
            aliases = {
                norm(player.get("web_name")),
                norm(f"{player.get('first_name', '')} {player.get('second_name', '')}"),
            } - {""}
            for alias in aliases:
                records = actual_by_alias.setdefault(alias, [])
                records.append(actual)

        fixture_rows = []
        all_rows = []
        for fixture in snapshot.get("predictions", []):
            home, away = fixture.get("home", "Home"), fixture.get("away", "Away")
            fixture_team_keys = team_aliases(home) | team_aliases(away)
            rows = []
            for predicted in fixture.get("players", []):
                prediction = predicted.get("prediction")
                if prediction is None:
                    continue
                alias = norm(predicted.get("name"))
                candidates = actual_by_alias.get(alias, [])
                unique = {row["id"]: row for row in candidates}
                candidates = list(unique.values())
                if len(candidates) > 1:
                    contextual = [row for row in candidates if team_aliases(row["team"]) & fixture_team_keys]
                    if len(contextual) == 1:
                        candidates = contextual
                if len(candidates) != 1:
                    continue
                actual = candidates[0]
                if actual["minutes"] <= 0:
                    continue
                predicted_points = float(prediction)
                row = {
                    "name": predicted.get("name") or actual["name"],
                    "rank": int(predicted.get("rank") or 0),
                    "prediction": round(predicted_points, 2),
                    "actual": round(actual["actual"], 2),
                    "difference": round(actual["actual"] - predicted_points, 2),
                    "minutes": actual["minutes"],
                }
                rows.append(row)
                all_rows.append(row)
            if rows:
                rows.sort(key=lambda row: (row["rank"] or 9999, -row["prediction"]))
                fixture_rows.append({"home": home, "away": away, "players": rows})

        if all_rows:
            errors = [abs(row["difference"]) for row in all_rows]
            bias = sum(row["difference"] for row in all_rows) / len(all_rows)
            payload["gameweeks"].append({
                "gameweek": gw,
                "available": True,
                "players_compared": len(all_rows),
                "mae": round(sum(errors) / len(errors), 2),
                "within_two": sum(error <= 2 for error in errors),
                "average_bias": round(bias, 2),
                "fixtures": fixture_rows,
            })

    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
