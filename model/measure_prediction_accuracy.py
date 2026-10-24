#!/usr/bin/env python3
"""Measure frozen ScoutXI predictions against completed official FPL results."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

FPL_API = "https://fantasy.premierleague.com/api"


def fetch_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "ScoutXI-Accuracy/1.0"})
    with urlopen(request, timeout=25) as response:
        return json.load(response)


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def unavailable(output: Path, reason: str, gameweek: int) -> None:
    output.write_text(
        json.dumps({"available": False, "gameweek": gameweek, "reason": reason}, indent=2) + "\n",
        encoding="utf-8",
    )


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
        unavailable(args.output, "no published snapshot", args.gameweek)
        return

    try:
        bootstrap = fetch_json(f"{FPL_API}/bootstrap-static/")
        event = next(
            (item for item in bootstrap.get("events", [])
             if int(item.get("id", -1)) == args.gameweek),
            None,
        )
        if event is None or not event.get("finished"):
            unavailable(args.output, "gameweek not marked finished by the official FPL API", args.gameweek)
            return
        live = fetch_json(f"{FPL_API}/event/{args.gameweek}/live/")
    except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
        unavailable(args.output, f"official FPL results temporarily unavailable ({type(error).__name__})", args.gameweek)
        return

    elements = {
        int(player["id"]): player
        for player in bootstrap.get("elements", [])
        if player.get("id") is not None
    }
    teams = {int(team["id"]): team.get("name", "Unknown") for team in bootstrap.get("teams", [])}
    actual_by_name: dict[str, dict] = {}
    for item in live.get("elements", []):
        player = elements.get(int(item.get("id", -1)))
        stats = item.get("stats", {})
        if not player or stats.get("total_points") is None:
            continue
        actual = {
            "points": float(stats["total_points"]),
            "position": int(player.get("element_type", 0)),
            "team": teams.get(int(player.get("team", 0)), "Unknown"),
        }
        aliases = {
            player.get("web_name", ""),
            f"{player.get('first_name', '')} {player.get('second_name', '')}".strip(),
        }
        for alias in aliases:
            key = normalize_name(alias)
            if key:
                if key in actual_by_name and actual_by_name[key]["points"] != actual["points"]:
                    actual_by_name[key] = {"points": float("nan"), "position": 0, "team": "Ambiguous"}
                else:
                    actual_by_name[key] = actual

    position_names = {1: "Goalkeeper", 2: "Defender", 3: "Midfielder", 4: "Forward"}
    rows = []
    unmatched = []
    for fixture in snapshot.get("predictions", []):
        for player in fixture.get("players", []):
            name = str(player.get("name", "")).strip()
            actual_record = actual_by_name.get(normalize_name(name))
            if actual_record is None or not np.isfinite(actual_record["points"]):
                unmatched.append(name)
                continue
            prediction = float(player["prediction"])
            actual = float(actual_record["points"])
            expected_minutes = player.get("expected_minutes")
            confidence = player.get("confidence", "unknown")
            rows.append({
                "name": name,
                "team": actual_record["team"],
                "position": position_names.get(actual_record["position"], "Unknown"),
                "prediction": prediction,
                "actual": actual,
                "error": round(prediction - actual, 3),
                "absolute_error": round(abs(prediction - actual), 3),
                "expected_minutes": float(expected_minutes) if expected_minutes is not None else None,
                "confidence": str(confidence),
                "p10": float(player["prediction_p10"]) if player.get("prediction_p10") is not None else None,
                "p90": float(player["prediction_p90"]) if player.get("prediction_p90") is not None else None,
            })

    if not rows:
        unavailable(args.output, "no prediction names matched official FPL player records", args.gameweek)
        return

    frame = pd.DataFrame(rows)
    errors = frame["prediction"] - frame["actual"]
    coverage = [
        row["p10"] <= row["actual"] <= row["p90"]
        for row in rows
        if row["p10"] is not None and row["p90"] is not None
    ]

    def summarize(group: pd.DataFrame, label: str, value: str) -> dict:
        group_errors = group["prediction"] - group["actual"]
        return {
            label: value,
            "players": int(len(group)),
            "mae": round(float(group_errors.abs().mean()), 3),
            "rmse": round(float(np.sqrt((group_errors ** 2).mean())), 3),
            "mean_error": round(float(group_errors.mean()), 3),
        }

    by_position = [
        summarize(group, "position", str(key))
        for key, group in frame.groupby("position", dropna=False)
    ]
    by_confidence = [
        summarize(group, "confidence", str(key))
        for key, group in frame.groupby("confidence", dropna=False)
    ]
    minute_bands = []
    for label, lower, upper in [
        ("0-29", 0, 30), ("30-59", 30, 60), ("60-89", 60, 90), ("90+", 90, float("inf"))
    ]:
        group = frame[frame["expected_minutes"].notna() &
                      (frame["expected_minutes"] >= lower) &
                      (frame["expected_minutes"] < upper)]
        if not group.empty:
            minute_bands.append(summarize(group, "expected_minutes_band", label))

    worst_errors = sorted(rows, key=lambda row: row["absolute_error"], reverse=True)[:10]
    result = {
        "available": True,
        "source": "official_fpl_api",
        "gameweek": args.gameweek,
        "players_evaluated": int(len(frame)),
        "players_unmatched": len(unmatched),
        "mae": round(float(errors.abs().mean()), 3),
        "rmse": round(float(np.sqrt((errors ** 2).mean())), 3),
        "mean_error": round(float(errors.mean()), 3),
        "p10_p90_coverage": round(float(np.mean(coverage)), 3) if coverage else None,
        "diagnostics": {
            "by_position": by_position,
            "by_confidence": by_confidence,
            "by_expected_minutes": minute_bands,
            "largest_absolute_errors": worst_errors,
        },
        "note": "Measured against frozen top-three predictions per fixture matched to official FPL points. Segment metrics are descriptive and should not drive model changes until several gameweeks provide adequate samples; this is not full-squad accuracy.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
