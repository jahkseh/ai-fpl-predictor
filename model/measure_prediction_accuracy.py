#!/usr/bin/env python3
"""Measure realized accuracy for frozen pre-match FPL predictions.

The external source may expose FPL's web_name instead of a generic name column.
If realized results are missing, publish an explicit unavailable status rather
than inventing metrics or failing the entire data-refresh workflow.
"""
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


def unavailable(reason: str, gameweek: int) -> dict:
    return {"available": False, "gameweek": gameweek, "reason": reason}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--gameweek", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    history = json.loads(args.history.read_text(encoding="utf-8"))
    snapshot = next(
        (item for item in history.get("gameweeks", [])
         if int(item.get("gameweek", -1)) == args.gameweek),
        None,
    )
    if snapshot is None:
        result = unavailable("no frozen prediction snapshot for this Gameweek", args.gameweek)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return

    try:
        stats = pd.read_csv(CURRENT_BASE.format(gw=args.gameweek))
    except Exception as exc:
        result = unavailable(f"realized results could not be loaded ({type(exc).__name__})", args.gameweek)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return

    if "total_points" not in stats.columns:
        result = unavailable("results source has no total_points column", args.gameweek)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return

    if "name" in stats.columns:
        name_column = "name"
    elif "web_name" in stats.columns:
        name_column = "web_name"
    elif {"first_name", "second_name"}.issubset(stats.columns):
        stats = stats.copy()
        stats["__player_name"] = (
            stats["first_name"].fillna("").astype(str).str.strip() + " " +
            stats["second_name"].fillna("").astype(str).str.strip()
        ).str.strip()
        name_column = "__player_name"
    else:
        result = unavailable("results source has no supported player-name column", args.gameweek)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return

    stats = stats.dropna(subset=[name_column, "total_points"])
    if stats.empty:
        result = unavailable("results source contains no realized player rows yet", args.gameweek)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return

    actual = (
        stats[[name_column, "total_points"]]
        .drop_duplicates(name_column)
        .set_index(name_column)["total_points"]
        .to_dict()
    )

    rows = []
    for fixture in snapshot.get("predictions", []):
        for player in fixture.get("players", []):
            name = str(player.get("name", "")).strip()
            if name not in actual:
                continue
            try:
                prediction = float(player["prediction"])
                realized = float(actual[name])
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isfinite(prediction) or not np.isfinite(realized):
                continue
            p10 = player.get("prediction_p10")
            p90 = player.get("prediction_p90")
            try:
                p10 = float(p10) if p10 is not None else None
                p90 = float(p90) if p90 is not None else None
            except (TypeError, ValueError):
                p10 = p90 = None
            rows.append({"name": name, "prediction": prediction, "actual": realized,
                         "p10": p10, "p90": p90})

    if not rows:
        result = unavailable(
            "no frozen top-three picks matched realized player names; check source naming",
            args.gameweek,
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return

    frame = pd.DataFrame(rows)
    errors = frame["prediction"] - frame["actual"]
    covered = [
        row["p10"] <= row["actual"] <= row["p90"]
        for row in rows
        if row["p10"] is not None and row["p90"] is not None
        and np.isfinite(row["p10"]) and np.isfinite(row["p90"])
        and row["p10"] <= row["p90"]
    ]
    result = {
        "available": True,
        "gameweek": args.gameweek,
        "players_evaluated": int(len(frame)),
        "mae": round(float(errors.abs().mean()), 3),
        "rmse": round(float(np.sqrt((errors ** 2).mean())), 3),
        "mean_error": round(float(errors.mean()), 3),
        "p10_p90_coverage": round(float(np.mean(covered)), 3) if covered else None,
        "note": "Only frozen top-three picks per fixture are evaluated; this is not full-squad accuracy.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
