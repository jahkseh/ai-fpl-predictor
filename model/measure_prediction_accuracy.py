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
        args.output.write_text(json.dumps({"available": False, "reason": "no published snapshot"}) + "\n")
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
    for fixture in snapshot["predictions"]:
        for player in fixture["players"]:
            name = player["name"]
            if name not in actual:
                continue
            rows.append({
                "name": name,
                "prediction": float(player["prediction"]),
                "actual": float(actual[name]),
                "p10": float(player["prediction_p10"]) if player.get("prediction_p10") is not None else None,
                "p90": float(player["prediction_p90"]) if player.get("prediction_p90") is not None else None,
            })

    if not rows:
        args.output.write_text(json.dumps({"available": False, "reason": "no matching realized players"}) + "\n")
        return

    frame = pd.DataFrame(rows)
    errors = frame["prediction"] - frame["actual"]
    coverage = [
        p["p10"] <= p["actual"] <= p["p90"]
        for p in rows
        if p["p10"] is not None and p["p90"] is not None
    ]

    result = {
        "available": True,
        "gameweek": args.gameweek,
        "players_evaluated": int(len(frame)),
        "mae": round(float(errors.abs().mean()), 3),
        "rmse": round(float(np.sqrt((errors ** 2).mean())), 3),
        "mean_error": round(float(errors.mean()), 3),
        "p10_p90_coverage": round(float(np.mean(coverage)), 3) if coverage else None,
        "note": "Measured only against the published top-three predictions per fixture; this is not full-squad accuracy.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
