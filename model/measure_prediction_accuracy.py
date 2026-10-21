#!/usr/bin/env python3
"""Measure realized accuracy for published gameweek predictions when results exist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_BASE = (
    "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/"
    "data/2026-2027/By%20Gameweek/GW{gw}/player_gameweek_stats.csv"
)


def snapshot_players(snapshot: dict) -> tuple[list[dict], str]:
    """Return the widest available snapshot and identify legacy limited data."""
    all_players = snapshot.get("all_players")
    if isinstance(all_players, list) and all_players:
        return all_players, "full_player_pool"

    legacy = [
        player
        for fixture in snapshot.get("predictions", [])
        for player in fixture.get("players", [])
    ]
    return legacy, "top_three_per_fixture_legacy"


def calculate_metrics(rows: list[dict], gameweek: int, scope: str) -> dict:
    """Calculate metrics from already matched prediction/result rows."""
    frame = pd.DataFrame(rows)
    errors = frame["prediction"] - frame["actual"]
    coverage = [
        row["p10"] <= row["actual"] <= row["p90"]
        for row in rows
        if row.get("p10") is not None and row.get("p90") is not None
    ]
    return {
        "available": True,
        "gameweek": gameweek,
        "coverage_scope": scope,
        "players_evaluated": int(len(frame)),
        "mae": round(float(errors.abs().mean()), 3),
        "rmse": round(float(np.sqrt((errors ** 2).mean())), 3),
        "mean_error": round(float(errors.mean()), 3),
        "p10_p90_coverage": round(float(np.mean(coverage)), 3) if coverage else None,
        "note": (
            "Metrics use the full saved player pool."
            if scope == "full_player_pool"
            else "Legacy snapshot: metrics cover only the top three predictions per fixture, not the full player pool."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--gameweek", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    history = json.loads(args.history.read_text(encoding="utf-8"))
    snapshot = next(
        (item for item in history.get("gameweeks", []) if int(item.get("gameweek", -1)) == args.gameweek),
        None,
    )
    if snapshot is None:
        args.output.write_text(
            json.dumps({"available": False, "reason": "no published snapshot"}) + "\n",
            encoding="utf-8",
        )
        return

    candidates, scope = snapshot_players(snapshot)
    if not candidates:
        args.output.write_text(
            json.dumps({"available": False, "reason": "snapshot contains no player predictions", "coverage_scope": scope}) + "\n",
            encoding="utf-8",
        )
        return

    stats = pd.read_csv(CURRENT_BASE.format(gw=args.gameweek))
    if "name" not in stats.columns or "total_points" not in stats.columns:
        raise ValueError("Current gameweek stats do not contain name/total_points")

    actual = (
        stats[["name", "total_points"]]
        .dropna(subset=["name"])
        .drop_duplicates("name")
        .set_index("name")["total_points"]
        .to_dict()
    )

    rows = []
    for player in candidates:
        name = str(player.get("name", "")).strip()
        if not name or name not in actual or player.get("prediction") is None:
            continue
        rows.append({
            "name": name,
            "prediction": float(player["prediction"]),
            "actual": float(actual[name]),
            "p10": float(player["prediction_p10"]) if player.get("prediction_p10") is not None else None,
            "p90": float(player["prediction_p90"]) if player.get("prediction_p90") is not None else None,
        })

    if not rows:
        args.output.write_text(
            json.dumps({
                "available": False,
                "reason": "no matching realized players",
                "coverage_scope": scope,
                "prediction_candidates": len(candidates),
                "matched_players": 0,
                "match_rate": 0,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        return

    result = calculate_metrics(rows, args.gameweek, scope)
    result["prediction_candidates"] = len(candidates)
    result["matched_players"] = len(rows)
    result["match_rate"] = round(len(rows) / len(candidates), 3)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
