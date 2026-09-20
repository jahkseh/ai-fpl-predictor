#!/usr/bin/env python3
"""Build a public, current FPL intelligence dataset for the website.

This script deliberately uses only published FPL-derived CSV data. It does not
invent missing values. The target gameweek is the next gameweek being prepared.
"""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from urllib.parse import quote

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


TEAM_CODES = {
    "Arsenal": "ARS", "Bournemouth": "BOU", "Brentford": "BRE",
    "Brighton": "BHA", "Chelsea": "CHE", "Coventry City": "COV",
    "Crystal Palace": "CRY", "Everton": "EVE", "Fulham": "FUL",
    "Hull City": "HUL", "Ipswich Town": "IPS", "Leeds": "LEE",
    "Liverpool": "LIV", "Man City": "MCI", "Man Utd": "MUN",
    "Newcastle": "NEW", "Nott'm Forest": "NFO", "Sunderland": "SUN",
    "Spurs": "TOT", "Aston Villa": "AVL",
}

TEAM_ACCENTS = {
    "Arsenal": "#b3121d", "Bournemouth": "#b3121d", "Brentford": "#c8102e",
    "Brighton": "#0057b8", "Chelsea": "#034694", "Coventry City": "#2f6fae",
    "Crystal Palace": "#1b458f", "Everton": "#003399", "Fulham": "#111111",
    "Hull City": "#a56b00", "Ipswich Town": "#0057b8", "Leeds": "#1d428a",
    "Liverpool": "#c8102e", "Man City": "#1c2c5b", "Man Utd": "#b91c1c",
    "Newcastle": "#111111", "Nott'm Forest": "#c81d25", "Sunderland": "#c8102e",
    "Spurs": "#132257", "Aston Villa": "#670e36",
}


def validate_differentials(differentials: list[dict]) -> None:
    if len(differentials) != 5:
        raise AssertionError(f"Expected exactly 5 differential picks, found {len(differentials)}")
    seen = set()
    required = {"name", "team", "position", "ownership", "prediction", "differential_score", "price", "fixture"}
    for pick in differentials:
        missing = required - set(pick)
        if missing:
            raise AssertionError(f"Differential pick missing fields: {sorted(missing)}")
        fixture = pick["fixture"]
        identity = (pick["name"], pick["team"], fixture.get("match_id"))
        if identity in seen:
            raise AssertionError(f"Duplicate differential pick: {identity}")
        seen.add(identity)
        values = np.array([float(pick["ownership"]), float(pick["prediction"]), float(pick["differential_score"])])
        if not np.isfinite(values).all():
            raise AssertionError(f"Non-finite differential values for {pick['name']}")
        if not 0 <= float(pick["ownership"]) <= 10:
            raise AssertionError(f"Differential ownership outside 0-10% for {pick['name']}")
        if not fixture.get("home") or not fixture.get("away") or not fixture.get("match_id"):
            raise AssertionError(f"Incomplete fixture for differential pick {pick['name']}")


def render_differential_cards(differentials: list[dict]) -> str:
    cards = []
    for index, pick in enumerate(differentials, start=1):
        team = pick["team"] or ""
        accent = TEAM_ACCENTS.get(team, "#27d17f")
        team_code = TEAM_CODES.get(team, "FPL")
        ownership = float(pick["ownership"])
        prediction = float(pick["prediction"])
        price = float(pick["price"])
        score = float(pick["differential_score"])
        fixture = pick.get("fixture") or {}
        fixture_text = (
            f"{fixture['home']} {'vs' if fixture.get('is_home') else 'at'} {fixture['away']}"
            if fixture.get("home") and fixture.get("away") else "Next fixture unavailable"
        )
        cards.append(
            f'<a class="differential-card" style="--differential-team-colour:{accent}" '
            f'href="player.html?player={quote(str(pick["name"]))}">'
            f'<span class="differential-rank">0{index}</span><div>'
            f'<div class="differential-card-head"><div><strong>{escape(str(pick["name"]))}</strong>'
            f'<span class="differential-meta">{escape(str(pick.get("position") or "Player"))} · {escape(team)}</span></div>'
            f'<span class="differential-team">{escape(team_code)}</span></div>'
            f'<div class="differential-stats">'
            f'<div class="differential-stat"><span>Ownership</span><strong>{ownership:.1f}%</strong></div>'
            f'<div class="differential-stat"><span>Projection</span><strong>{prediction:.2f}</strong></div>'
            f'<div class="differential-stat"><span>Price</span><strong>£{price:.1f}</strong></div></div>'
            f'<p class="differential-fixture">{escape(fixture_text)}</p>'
            f'<div class="differential-score">Differential score · {score:.2f}</div></div></a>'
        )
    return "\n            ".join(cards)


def update_index_differentials(index_path: Path, differentials: list[dict]) -> None:
    if not index_path.exists():
        raise FileNotFoundError(f"Website index not found: {index_path}")
    html = index_path.read_text(encoding="utf-8")
    start_marker = "            <!-- DIFFERENTIAL_CARDS_START -->"
    end_marker = "            <!-- DIFFERENTIAL_CARDS_END -->"
    start = html.find(start_marker)
    end = html.find(end_marker)
    if start < 0 or end < 0 or end < start:
        raise ValueError("Differential card markers are missing or out of order in index.html")
    replacement = start_marker + "\n            " + render_differential_cards(differentials) + "\n            " + end_marker
    html = html[:start] + replacement + html[end + len(end_marker):]
    index_path.write_text(html, encoding="utf-8")


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

    # Predictions are keyed by player + fixture, not player name alone.
    # This matters because FPL display names are not guaranteed to be unique
    # (for example, there can be more than one player named "Palmer").
    predictions_by_fixture = {}
    if args.predictions.exists():
        raw = json.loads(args.predictions.read_text(encoding="utf-8"))
        for fixture_prediction in raw.get("predictions", []):
            fixture_home = fixture_prediction.get("home")
            fixture_away = fixture_prediction.get("away")
            for player in fixture_prediction.get("players", []):
                canonical_name = NAME_ALIASES.get(player["name"], player["name"])
                predictions_by_fixture[
                    (canonical_name, fixture_home, fixture_away)
                ] = float(player["prediction"])

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
        canonical_name = NAME_ALIASES.get(name, name)
        prediction = None
        if fixture:
            prediction = predictions_by_fixture.get(
                (canonical_name, fixture["home"], fixture["away"])
            )
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
            "prediction": prediction,
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

    validate_differentials([
        {k: r.get(k) for k in [
            "name", "team", "position", "ownership", "prediction",
            "differential_score", "price", "fixture"
        ]}
        for r in differentials
    ])

    differential_payload = {
        "season": "2026/27",
        "data_gameweek": latest_gw,
        "next_gameweek": args.gameweek,
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "picks": [
            {k: r.get(k) for k in [
                "name", "team", "position", "ownership", "prediction",
                "differential_score", "price", "form", "fixture"
            ]}
            for r in differentials
        ],
    }

    output = {
        "season": "2026/27",
        "data_gameweek": latest_gw,
        "next_gameweek": args.gameweek,
        "generated_at_utc": differential_payload["generated_at_utc"],
        "player_count": len(player_rows),
        "fixtures": fixture_output,
        "differentials": differential_payload["picks"],
        "players": player_rows,
    }
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Keep a small, dedicated payload for the website's Differential Picks section.
    # The main live_data.json remains the canonical player dataset.
    differential_output = args.output.with_name("differentials.json")
    differential_output.write_text(
        json.dumps(differential_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Keep the published page usable even if a browser/CDN blocks JSON fetches.
    # The same generated picks are rendered directly into index.html, while the
    # JavaScript loader remains available to refresh the cards when JSON loads.
    update_index_differentials(Path("index.html"), differential_payload["picks"])

    print(f"Wrote {args.output} with {len(player_rows)} players; GW{latest_gw} data / GW{args.gameweek} fixtures.")

if __name__ == "__main__":
    main()
