#!/usr/bin/env python3
"""Append available completed-gameweek accuracy results without overwriting history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--history", required=True, type=Path)
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    if not result.get("available"):
        print("No realized accuracy result; keeping the existing trend history unchanged.")
        return

    required = ("gameweek", "players_evaluated", "mae", "rmse", "mean_error")
    missing = [key for key in required if result.get(key) is None]
    if missing:
        raise ValueError(f"Available accuracy result is missing fields: {missing}")

    if args.history.exists():
        history = json.loads(args.history.read_text(encoding="utf-8"))
    else:
        history = {"season": "2026/27", "gameweeks": []}

    record = {key: result.get(key) for key in (
        "gameweek", "players_evaluated", "players_unmatched", "mae", "rmse",
        "mean_error", "p10_p90_coverage", "source", "note"
    ) if key in result}
    records = [
        item for item in history.get("gameweeks", [])
        if int(item.get("gameweek", -1)) != int(record["gameweek"])
    ]
    records.append(record)
    records.sort(key=lambda item: int(item["gameweek"]))
    history["season"] = history.get("season", "2026/27")
    history["gameweeks"] = records[-38:]
    history["gameweek_count"] = len(history["gameweeks"])
    args.history.write_text(json.dumps(history, indent=2) + "\\n", encoding="utf-8")
    print(f"Published accuracy trend for {len(history['gameweeks'])} gameweeks.")


if __name__ == "__main__":
    main()
