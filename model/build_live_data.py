#!/usr/bin/env python3
"""Build a public, current FPL intelligence dataset for the website.

This script deliberately uses only published FPL-derived CSV data. It does not
invent missing values. The target gameweek is the next gameweek being prepared.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = (
    "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/"
    "data/2026-2027/By%20Gameweek/GW{gw}/"
)

SLUGS = {
    "everton": "Everton", "ipswich-town": "Ipswich Town",
    "nottingham-forest": "Nott'm Forest", "coventry-city": "Coventry City",
    "leeds-united": "Leeds", "crystal-palace": "Crystal Palace",
    "manchester-city": "Man City", "sunderland": "Sunderland",
    "afc-bournemouth": "Bournemouth", "liverpool": "Liverpool",
    "fulham": "Fulham", "manchester-united": "Man Utd",
    "brighton-hove-albion": "Brighton", "arsenal": "Arsenal",
    "tottenham-hotspur": "Spurs", "aston-villa": "Aston Villa",
    "newcastle-united": "Newcastle", "hull-city": "Hull City",
    "brentford": "Brentford", "chelsea": "Chelsea",
}

NAME_ALIASES = {"B.Fernandes": "B. Fernandes"}

def read_csv(gw: int, filename: str) -> pd.DataFrame:
    url = BASE.format(gw=gw) + filename
    df = pd.read_csv(url)
    if df.empty:
        raise ValueError(f"Empty source: {url}")
    return df

def fixture_names(match_id: str) -> tuple[str, str]:
    parts = str(match_id).replace("26-27-prem-", "").split("-vs-")
    if len(parts) != 2 or parts[0] not in SLUGS or parts[1] not in SLUGS:
        raise ValueError(f"Unknown fixture slug: {match_id}")
    return SLUGS[parts[0]], SLUGS[parts[1]]

def safe_float(value):
    if pd.isna(value):
        return None
    value = float(value)
    return value if np.isfinite(value) else None

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameweek", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("live_data.json"))
    parser.add_argument("--predictions", type=Path, default=Path("predictions.json"))
    args = parser.parse_args()

    if not 2 <= args.gameweek <= 38:
        raise SystemExit("--gameweek must be between 2 and 38")

    latest_gw = args.gameweek - 1
    current = pd.concat(
        [read_csv(gw, "player_gameweek_stats.csv").assign(gw=gw)
         for gw in range(1, latest_gw + 1)],
        ignore_index=True,
    )
    players = read_csv(args.gameweek, "players.csv")
    teams = read_csv(args.gameweek, "teams.csv")
    fixtures = read_csv(args.gameweek, "fixtures.csv")
    fixtures = fixtures[
        (fixtures["tournament"] == "prem")
        & (fixtures["gameweek"] == args.gameweek)
    ].copy()

    if len(fixtures) != 10:
        raise AssertionError(f"Expected 10 PL fixtures, found {len(fixtures)}")

    team_map = teams.set_index("code")["name"].to_dict()
    position_map = (
        players[["player_id", "position"]]
        .drop_duplicates("player_id")
        .set_index("player_id")["position"]
        .to_dict()
    )

    current = current.sort_values(["id", "gw"])
    for col in ["total_points", "minutes", "expected_goals", "expected_assists", "starts", "goals_scored", "assists", "bps"]:
        current[f"avg3_{col}"] = current.groupby("id")[col].transform(
            lambda s: s.shift(0).rolling(3, min_periods=1).mean()
        )

    latest = current[current["gw"] == latest_gw].copy()
    latest = latest.drop_duplicates("id", keep="last")

    # Match each team to its next fixture.
    next_fixture = {}
    for _, row in fixtures.iterrows():
        home, away = fixture_names(row["match_id"])
        home_code = int(row["home_team"])
        away_code = int(row["away_team"])
        next_fixture[home_code] = {
            "home": home, "away": away, "is_home": True,
            "kickoff": row["kickoff_time"], "match_id": row["match_id"],
        }
        next_fixture[away_code] = {
            "home": home, "away": away, "is_home": False,
            "kickoff": row["kickoff_time"], "match_id": row["match_id"],
        }

    predictions = {}
    if args.predictions.exists():
        raw = json.loads(args.predictions.read_text(encoding="utf-8"))
        for fixture in raw.get("predictions", []):
            for player in fixture.get("players", []):
                predictions[NAME_ALIASES.get(player["name"], player["name"])] = float(player["prediction"])

    rows = []
    for _, row in latest.iterrows():
        player_id = int(row["id"])
        team_code = int(row["team_code"]) if "team_code" in row and pd.notna(row["team_code"]) else None
        # player_gameweek_stats already carries team_code in most snapshots; use players.csv as fallback.
        if team_code is None:
            match = players[players["player_id"] == player_id]
            team_code = int(match.iloc[0]["team_code"]) if not match.empty else None
        fixture = next_fixture.get(team_code)

        name = str(row["web_name"])
        item = {
            "id": player_id,
            "name": NAME_ALIASES.get(name, name),
            "first_name": str(row.get("first_name", "")),
            "second_name": str(row.get("second_name", "")),
            "team": team_map.get(team_code),
            "position": position_map.get(player_id),
            "price": safe_float(row.get("now_cost")),
            "ownership": safe_float(row.get("selected_by_percent")),
            "form": safe_float(row.get("form")),
            "points_per_game": safe_float(row.get("points_per_game")),
            "total_points": int(row.get("total_points", 0)),
            "minutes": int(row.get("minutes", 0)),
            "starts": int(row.get("starts", 0)),
            "goals": int(row.get("goals_scored", 0)),
            "assists": int(row.get("assists", 0)),
            "clean_sheets": int(row.get("clean_sheets", 0)),
            "bonus": int(row.get("bonus", 0)),
            "bps": int(row.get("bps", 0)),
            "xg": safe_float(row.get("expected_goals")),
            "xa": safe_float(row.get("expected_assists")),
            "xgi": safe_float(row.get("expected_goal_involvements")),
            "xgc": safe_float(row.get("expected_goals_conceded")),
            "influence": safe_float(row.get("influence")),
            "creativity": safe_float(row.get("creativity")),
            "threat": safe_float(row.get("threat")),
            "ict_index": safe_float(row.get("ict_index")),
            "defensive_contribution": safe_float(row.get("defensive_contribution")),
            "expected_goals_per_90": safe_float(row.get("expected_goals_per_90")),
            "expected_assists_per_90": safe_float(row.get("expected_assists_per_90")),
            "expected_goal_involvements_per_90": safe_float(row.get("expected_goal_involvements_per_90")),
            "expected_goals_conceded_per_90": safe_float(row.get("expected_goals_conceded_per_90")),
            "saves_per_90": safe_float(row.get("saves_per_90")),
            "starts_per_90": safe_float(row.get("starts_per_90")),
            "defensive_contribution_per_90": safe_float(row.get("defensive_contribution_per_90")),
            "chance_next_round": safe_float(row.get("chance_of_playing_next_round")),
            "news": None if pd.isna(row.get("news")) else str(row.get("news")),
            "prediction": predictions.get(NAME_ALIASES.get(name, name)),
            "recent_3_points": safe_float(row.get("avg3_total_points")),
            "recent_3_minutes": safe_float(row.get("avg3_minutes")),
            "recent_3_xg": safe_float(row.get("avg3_expected_goals")),
            "recent_3_xa": safe_float(row.get("avg3_expected_assists")),
            "recent_3_starts": safe_float(row.get("avg3_starts")),
            "recent_3_goals": safe_float(row.get("avg3_goals_scored")),
            "recent_3_assists": safe_float(row.get("avg3_assists")),
            "recent_3_bps": safe_float(row.get("avg3_bps")),
            "fixture": None,
        }
        if fixture:
            item["fixture"] = {
                "home": fixture["home"], "away": fixture["away"],
                "is_home": fixture["is_home"], "kickoff": fixture["kickoff"],
                "match_id": fixture["match_id"],
            }
        rows.append(item)

    player_rows = [r for r in rows if r["team"]]
    predicted = [r for r in player_rows if r["prediction"] is not None]
    differentials = sorted(
        [r for r in predicted if r["ownership"] is not None and r["ownership"] <= 10],
        key=lambda r: (r["prediction"] * (1 - r["ownership"] / 100)),
        reverse=True,
    )[:5]
    for r in differentials:
        r["differential_score"] = round(r["prediction"] * (1 - r["ownership"] / 100), 3)

    fixture_output = []
    for _, row in fixtures.sort_values("kickoff_time").iterrows():
        home, away = fixture_names(row["match_id"])
        fixture_output.append({
            "home": home, "away": away,
            "kickoff": row["kickoff_time"],
            "match_id": row["match_id"],
            "home_elo": safe_float(row.get("home_team_elo")),
            "away_elo": safe_float(row.get("away_team_elo")),
            "status": "finished" if bool(row.get("finished")) else "scheduled",
            "home_xg": safe_float(row.get("home_expected_goals_xg")),
            "away_xg": safe_float(row.get("away_expected_goals_xg")),
        })

    output = {
        "season": "2026/27",
        "data_gameweek": latest_gw,
        "next_gameweek": args.gameweek,
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "player_count": len(player_rows),
        "fixtures": fixture_output,
        "differentials": [
            {k: r.get(k) for k in [
                "name", "team", "position", "ownership", "prediction",
                "differential_score", "price", "form", "fixture"
            ]}
            for r in differentials
        ],
        "players": player_rows,
    }
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output} with {len(player_rows)} players; GW{latest_gw} data / GW{args.gameweek} fixtures.")

if __name__ == "__main__":
    main()
