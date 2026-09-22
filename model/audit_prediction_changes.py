#!/usr/bin/env python3
"""Compare newly generated predictions with the previously published model output.

This is an audit trail, not a claim of causal feature importance. It records
prediction changes so the site/model can be inspected after each refresh.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def player_map(payload: dict) -> dict[str, dict]:
    return {
        str(player["name"]): player
        for player in payload.get("all_players", [])
        if player.get("name") is not None and player.get("prediction") is not None
    }


def build_audit(previous: dict | None, current: dict) -> dict:
    current_players = player_map(current)
    previous_players = player_map(previous or {})

    changes = []
    for name, player in current_players.items():
        old = previous_players.get(name)
        if not old:
            continue
        old_prediction = float(old["prediction"])
        new_prediction = float(player["prediction"])
        changes.append({
            "name": name,
            "old_prediction": round(old_prediction, 3),
            "new_prediction": round(new_prediction, 3),
            "change": round(new_prediction - old_prediction, 3),
            "old_confidence": old.get("confidence"),
            "new_confidence": player.get("confidence"),
            "expected_minutes": player.get("expected_minutes"),
            "fixture_difficulty": player.get("fixture_difficulty"),
        })

    changes.sort(key=lambda row: abs(row["change"]), reverse=True)
    return {
        "season": current.get("season"),
        "gameweek": current.get("gameweek"),
        "comparison_available": previous is not None and bool(previous_players),
        "players_compared": len(changes),
        "largest_changes": changes[:25],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    previous = load(args.previous) if args.previous and args.previous.exists() else None
    audit = build_audit(previous, load(args.current))
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
