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
    actual_by_name: dict[str, float] = {}
    for item in live.get("elements", []):
        player = elements.get(int(item.get("id", -1)))
        stats = item.get("stats", {})
        if not player or stats.get("total_points") is None:
            continue
        points = float(stats["total_points"])
        aliases = {
            player.get("web_name", ""),
            f"{player.get('first_name', '')} {player.get('second_name', '')}".strip(),
        }
        for alias in aliases:
            key = normalize_name(alias)
            if key:
                if key in actual_by_name and actual_by_name[key] != points:
                    actual_by_name[key] = float("nan")
                else:
                    actual_by_name[key] = points

    rows = []
    unmatched = []
    for fixture in snapshot.get("predictions", []):
        for player in fixture.get("players", []):
            name = str(player.get("name", "")).strip()
            actual = actual_by_name.get(normalize_name(name))
            if actual is None or not np.isfinite(actual):
                unmatched.append(name)
                continue
            rows.append({
                "name": name,
                "prediction": float(player["prediction"]),
                "actual": float(actual),
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
        "note": "Measured against frozen top-three predictions per fixture matched to official FPL points. This is not full-squad accuracy.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
