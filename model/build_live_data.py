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
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from historical_intelligence import load_completed_seasons, normalize_name

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
    """Render compact differential cards for the static homepage."""
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
            f'<span class="differential-rank">0{index}</span>'
            f'<div class="differential-card-content">'
            f'<div class="differential-card-head"><div><strong>{escape(str(pick["name"]))}</strong>'
            f'<span class="differential-meta">{escape(str(pick.get("position") or "Player"))} · {escape(team)}</span></div>'
            f'<span class="differential-team">{escape(team_code)}</span></div>'
            f'<div class="differential-projection"><span>Model projection</span><strong>{prediction:.2f}</strong></div>'
            f'<div class="differential-secondary"><span>{ownership:.1f}% owned</span><span>£{price:.1f}</span></div>'
            f'<p class="differential-fixture">{escape(fixture_text)}</p>'
            f'<div class="differential-score">Differential score · {score:.2f}</div>'
            f'</div></a>'
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

def load_official_fpl_fixture_difficulties() -> dict[tuple[int, str, str], tuple[int, int]]:
    """Fetch official FPL home/away FDR values keyed by gameweek and fixture names.

    Returns an empty mapping if the official API is temporarily unavailable; callers
    can then retain the clearly labelled model-based estimate as a fallback.
    """
    try:
        headers = {"User-Agent": "AI-FPL-Predictor/1.0 (public fixture difficulty lookup)"}

        def fetch_json(url: str):
            request = Request(url, headers=headers)
            with urlopen(request, timeout=12) as response:
                return json.loads(response.read().decode("utf-8"))

        bootstrap = fetch_json("https://fantasy.premierleague.com/api/bootstrap-static/")
        fixtures = fetch_json("https://fantasy.premierleague.com/api/fixtures/")
        api_names = {int(team["id"]): str(team["name"]) for team in bootstrap.get("teams", [])}
        canonical = {normalize_name(name): name for name in SLUGS.values()}
        aliases = {
            "nottinghamforest": "Nott'm Forest",
            "tottenhamhotspur": "Spurs",
            "brightonandhovealbion": "Brighton",
            "manchesterunited": "Man Utd",
            "manchestercity": "Man City",
            "afcbournemouth": "Bournemouth",
            "leedsunited": "Leeds",
        }
        for alias, display_name in aliases.items():
            canonical[normalize_name(alias)] = display_name

        result = {}
        for fixture in fixtures:
            event = fixture.get("event")
            home_name = canonical.get(normalize_name(api_names.get(fixture.get("team_h"), "")))
            away_name = canonical.get(normalize_name(api_names.get(fixture.get("team_a"), "")))
            home_rating = fixture.get("team_h_difficulty")
            away_rating = fixture.get("team_a_difficulty")
            if event is None or not home_name or not away_name:
                continue
            if home_rating not in (1, 2, 3, 4, 5) or away_rating not in (1, 2, 3, 4, 5):
                continue
            result[(int(event), home_name, away_name)] = (int(home_rating), int(away_rating))
        return result
    except Exception as exc:
        print(f"Official FPL fixture difficulty unavailable; using labelled model estimates: {exc}")
        return {}


def safe_float(value):
    if pd.isna(value):
        return None
    value = float(value)
    return value if np.isfinite(value) else None

def build_player_profiles(
    current: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    target_gameweek: int,
    prediction_details_by_fixture: dict,
    historical: pd.DataFrame | None = None,
) -> dict:
    """Build a deterministic player-profile dataset from published model inputs.

    This is presentation data only: it does not retrain the model or add new
    predictive features. Recent form is gameweek-level current-season data;
    model_inputs mirrors the feature groups exposed by prediction_engine.
    """
    if historical is None:
        historical = load_completed_seasons()

    historical = historical.copy()
    if "player_key" not in historical.columns:
        historical["player_key"] = historical["name"].map(normalize_name)

    team_map = teams.set_index("code")["name"].to_dict()
    position_map = (
        players[["player_id", "position"]]
        .drop_duplicates("player_id")
        .set_index("player_id")["position"]
        .to_dict()
    )
    player_team_map = (
        players[["player_id", "team_code"]]
        .drop_duplicates("player_id")
        .set_index("player_id")["team_code"]
        .to_dict()
    )
    team_strength = (
        teams[["code", "strength"]]
        .dropna(subset=["strength"])
        .drop_duplicates("code")
        .sort_values(["strength", "code"])
        .reset_index(drop=True)
    )
    difficulty_map = {}
    if len(team_strength) >= 5:
        difficulty_map = dict(zip(
            team_strength["code"].astype(int),
            np.ceil((np.arange(len(team_strength)) + 1) / len(team_strength) * 5)
            .clip(1, 5)
            .astype(int),
        ))
    official_fdr = load_official_fpl_fixture_difficulties()

    future_fixtures = []
    for gw in range(target_gameweek, min(target_gameweek + 4, 38) + 1):
        try:
            frame = read_csv(gw, "fixtures.csv")
        except (ValueError, pd.errors.EmptyDataError):
            continue
        frame = frame[
            (frame["tournament"] == "prem")
            & (frame["gameweek"] == gw)
        ].copy()
        for _, fixture in frame.sort_values("kickoff_time").iterrows():
            home, away = fixture_names(fixture["match_id"])
            home_code = int(fixture["home_team"])
            away_code = int(fixture["away_team"])
            official_pair = official_fdr.get((gw, home, away))
            home_difficulty = official_pair[0] if official_pair else difficulty_map.get(away_code)
            away_difficulty = official_pair[1] if official_pair else difficulty_map.get(home_code)
            difficulty_source = "official_fpl" if official_pair else "model_estimate"
            future_fixtures.extend([
                {
                    "gw": gw,
                    "team_code": home_code,
                    "opponent": away,
                    "is_home": True,
                    "home": home,
                    "away": away,
                    "kickoff": fixture["kickoff_time"],
                    "match_id": fixture["match_id"],
                    "difficulty": home_difficulty,
                    "difficulty_source": difficulty_source,
                },
                {
                    "gw": gw,
                    "team_code": away_code,
                    "opponent": home,
                    "is_home": False,
                    "home": home,
                    "away": away,
                    "kickoff": fixture["kickoff_time"],
                    "match_id": fixture["match_id"],
                    "difficulty": away_difficulty,
                    "difficulty_source": difficulty_source,
                },
            ])

    latest_gw = int(current["gw"].max()) if not current.empty else target_gameweek - 1
    latest = current[current["gw"] == latest_gw].drop_duplicates("id", keep="last")
    profiles = []

    for _, player in latest.iterrows():
        player_id = int(player["id"])
        team_code = (
            int(player["team_code"])
            if "team_code" in player and pd.notna(player["team_code"])
            else int(player_team_map[player_id])
            if player_id in player_team_map and pd.notna(player_team_map[player_id])
            else None
        )
        team = team_map.get(team_code)
        name = NAME_ALIASES.get(str(player["web_name"]), str(player["web_name"]))
        position = position_map.get(player_id)
        if not team:
            continue

        season_rows = current[current["id"] == player_id].sort_values("gw")
        recent_rows = season_rows.tail(5)
        season_minutes = float(pd.to_numeric(season_rows["minutes"], errors="coerce").fillna(0).sum())
        season_points = float(pd.to_numeric(season_rows["total_points"], errors="coerce").fillna(0).sum())
        season_xg = float(pd.to_numeric(season_rows["expected_goals"], errors="coerce").fillna(0).sum())
        season_xa = float(pd.to_numeric(season_rows["expected_assists"], errors="coerce").fillna(0).sum())
        season_starts = float(pd.to_numeric(season_rows["starts"], errors="coerce").fillna(0).sum())
        points_per_90 = season_points / season_minutes * 90 if season_minutes > 0 else None
        xg_per_90 = season_xg / season_minutes * 90 if season_minutes > 0 else None
        xa_per_90 = season_xa / season_minutes * 90 if season_minutes > 0 else None

        fixture = None
        team_fixtures = [f for f in future_fixtures if f["team_code"] == team_code]
        if team_fixtures:
            fixture = min(team_fixtures, key=lambda f: (f["gw"], f["kickoff"]))

        prediction_details = {}
        if fixture:
            key = (name, fixture["home"], fixture["away"])
            prediction_details = prediction_details_by_fixture.get(key, {})

        model_inputs = prediction_details.get("model_inputs", {})
        fixture_context = prediction_details.get("fixture_context", {})
        recent_form = []
        for _, gw_row in recent_rows.iterrows():
            gw = int(gw_row["gw"])
            gw_team_code = (
                int(gw_row["team_code"])
                if "team_code" in gw_row and pd.notna(gw_row["team_code"])
                else team_code
            )
            opponent_code = gw_row.get("opponent_team")
            opponent = team_map.get(int(opponent_code)) if pd.notna(opponent_code) else None
            was_home = bool(gw_row.get("was_home", False))
            recent_form.append({
                "gw": gw,
                "points": safe_float(gw_row.get("total_points")),
                "minutes": safe_float(gw_row.get("minutes")),
                "starts": safe_float(gw_row.get("starts")),
                "xg": safe_float(gw_row.get("expected_goals")),
                "xa": safe_float(gw_row.get("expected_assists")),
                "ict_index": safe_float(gw_row.get("ict_index")),
                "bps": safe_float(gw_row.get("bps")),
                "bonus": safe_float(gw_row.get("bonus")),
                "fixture": (
                    (team_map.get(gw_team_code, team) + " vs " + opponent)
                    if was_home and opponent
                    else ((team_map.get(gw_team_code, team) + " at " + opponent) if opponent else "GW" + str(gw))
                ),
                "is_home": was_home,
            })

        def fmt(value, suffix=""):
            return (f"{float(value):.2f}{suffix}" if value is not None and np.isfinite(float(value)) else "n/a")

        # Match current-season display names to historical full names robustly.
        # FPL web_name values can be shortened (e.g. "Rice" vs "Declan Rice"),
        # so prefer the current first/second-name combination when available.
        history_candidates = {
            normalize_name(name),
            normalize_name(str(player.get("first_name", "")) + " " + str(player.get("second_name", ""))),
        }
        history_candidates = {key for key in history_candidates if key and key != "nan"}
        player_history = historical[historical["player_key"].isin(history_candidates)].copy()
        history_seasons = []
        if not player_history.empty:
            grouped_history = (
                player_history.groupby(["season", "club_key", "team"], as_index=False)
                .agg(
                    minutes=("minutes", "sum"),
                    starts=("starts", "sum"),
                    points=("total_points", "sum"),
                    goals=("goals_scored", "sum"),
                    assists=("assists", "sum"),
                    xg=("expected_goals", "sum"),
                    xa=("expected_assists", "sum"),
                    ict_index=("ict_index", "sum"),
                    bps=("bps", "sum"),
                    bonus=("bonus", "sum"),
                )
                .sort_values(["season", "club_key"])
            )
            for _, history_row in grouped_history.iterrows():
                minutes = float(history_row["minutes"])
                history_seasons.append({
                    "season": str(history_row["season"]),
                    "club": str(history_row["team"]),
                    "minutes": int(minutes),
                    "starts": int(history_row["starts"]),
                    "points": int(history_row["points"]),
                    "goals": int(history_row["goals"]),
                    "assists": int(history_row["assists"]),
                    "xg": safe_float(history_row["xg"]),
                    "xa": safe_float(history_row["xa"]),
                    "ict_index": safe_float(history_row["ict_index"]),
                    "bps": safe_float(history_row["bps"]),
                    "bonus": safe_float(history_row["bonus"]),
                    "points_per_90": safe_float(history_row["points"] / minutes * 90 if minutes > 0 else None),
                    "xg_per_90": safe_float(history_row["xg"] / minutes * 90 if minutes > 0 else None),
                    "xa_per_90": safe_float(history_row["xa"] / minutes * 90 if minutes > 0 else None),
                    "ict_index_per_90": safe_float(history_row["ict_index"] / minutes * 90 if minutes > 0 else None),
                    "bps_per_90": safe_float(history_row["bps"] / minutes * 90 if minutes > 0 else None),
                    "bonus_per_90": safe_float(history_row["bonus"] / minutes * 90 if minutes > 0 else None),
                })

        career_minutes = int(sum(item["minutes"] for item in history_seasons))
        career_points = int(sum(item["points"] for item in history_seasons))
        career_goals = int(sum(item["goals"] for item in history_seasons))
        career_assists = int(sum(item["assists"] for item in history_seasons))
        career_xg = float(sum(item["xg"] or 0 for item in history_seasons))
        career_xa = float(sum(item["xa"] or 0 for item in history_seasons))
        career_ict = float(sum((item.get("ict_index") or 0) for item in history_seasons))
        career_bps = float(sum((item.get("bps") or 0) for item in history_seasons))
        career_bonus = float(sum((item.get("bonus") or 0) for item in history_seasons))
        career = {
            "minutes": career_minutes,
            "starts": int(sum(item["starts"] for item in history_seasons)),
            "points": career_points,
            "goals": career_goals,
            "assists": career_assists,
            "xg": safe_float(career_xg),
            "xa": safe_float(career_xa),
            "points_per_90": safe_float(career_points / career_minutes * 90 if career_minutes > 0 else None),
            "xg_per_90": safe_float(career_xg / career_minutes * 90 if career_minutes > 0 else None),
            "xa_per_90": safe_float(career_xa / career_minutes * 90 if career_minutes > 0 else None),
            "ict_index": safe_float(career_ict),
            "bps": safe_float(career_bps),
            "bonus": safe_float(career_bonus),
            "ict_index_per_90": safe_float(career_ict / career_minutes * 90 if career_minutes > 0 else None),
            "bps_per_90": safe_float(career_bps / career_minutes * 90 if career_minutes > 0 else None),
            "bonus_per_90": safe_float(career_bonus / career_minutes * 90 if career_minutes > 0 else None),
        }

        why = [
            {"label": "Historical ability", "value": fmt(model_inputs.get("historical_ability", {}).get("points_per_90"), " pts/90"),
             "detail": "Career historical points per 90 used as the long-term player prior."},
            {"label": "Current form", "value": fmt(model_inputs.get("current_form", {}).get("recent_3_points"), " pts/GW"),
             "detail": "Average points across the previous three completed gameweeks."},
            {"label": "Club history", "value": fmt(model_inputs.get("current_club", {}).get("points_per_90"), " pts/90"),
             "detail": "Most recent player-at-club historical rate; after a transfer this can refer to the previous club."},
            {"label": "Team attack", "value": fmt(model_inputs.get("team_attack", {}).get("xg_for_per_90"), " xG/90"),
             "detail": "Current team attacking environment from completed fixtures."},
            {"label": "Opponent defence", "value": fmt(model_inputs.get("opponent_defence", {}).get("xg_against_per_90"), " xGA/90"),
             "detail": "Opponent defensive environment from completed fixtures."},
            {"label": "Expected minutes", "value": fmt(model_inputs.get("expected_minutes"), " min"),
             "detail": "Separate expected-minutes model output."},
            {"label": "Home / away", "value": "Home" if model_inputs.get("home_away") == "home" else "Away",
             "detail": "Fixture location supplied to the points model."},
            {"label": "Fixture environment", "value": fmt(model_inputs.get("fixture_environment"), " combined xG"),
             "detail": "Team and opponent attacking/defensive fixture environment."},
        ]

        profiles.append({
            "id": player_id,
            "name": name,
            "position": position,
            "team": team,
            "price": safe_float(player.get("now_cost")) / 10 if pd.notna(player.get("now_cost")) else None,
            "ownership": safe_float(player.get("selected_by_percent")),
            "form": safe_float(player.get("form")),
            "prediction": safe_float(prediction_details.get("prediction")),
            "prediction_p10": safe_float(prediction_details.get("prediction_p10")),
            "prediction_p90": safe_float(prediction_details.get("prediction_p90")),
            "prediction_range": safe_float(prediction_details.get("prediction_range")),
            "confidence": prediction_details.get("confidence"),
            "expected_minutes": safe_float(prediction_details.get("expected_minutes")),
            "start_probability": safe_float(prediction_details.get("start_probability")),
            "sixty_probability": safe_float(prediction_details.get("sixty_probability")),
            "fixture": fixture,
            "model_inputs": model_inputs,
            "fixture_context": fixture_context,
            "why": why,
            "recent_form": recent_form,
            "underlying": {
                "xg_per_90": xg_per_90,
                "xa_per_90": xa_per_90,
                "points_per_90": points_per_90,
                "starts": int(season_starts),
                "minutes": int(season_minutes),
                "goals": int(pd.to_numeric(season_rows["goals_scored"], errors="coerce").fillna(0).sum()),
                "assists": int(pd.to_numeric(season_rows["assists"], errors="coerce").fillna(0).sum()),
                "ict_index_per_90": safe_float(model_inputs.get("current_form", {}).get("ict_index_per_90")),
                "bps_per_90": safe_float(model_inputs.get("current_form", {}).get("bps_per_90")),
                "bonus_per_90": safe_float(model_inputs.get("current_form", {}).get("bonus_per_90")),
                "influence_per_90": safe_float(model_inputs.get("current_form", {}).get("influence_per_90")),
                "creativity_per_90": safe_float(model_inputs.get("current_form", {}).get("creativity_per_90")),
                "threat_per_90": safe_float(model_inputs.get("current_form", {}).get("threat_per_90")),
                "xgi_per_90": safe_float(model_inputs.get("current_form", {}).get("xgi_per_90")),
            },
            "historical_record": {
                "career": career,
                "seasons": history_seasons,
            },
            "next_fixtures": [
                {
                    "gw": f["gw"],
                    "opponent": f["opponent"],
                    "is_home": f["is_home"],
                    "fixture": f["home"] + (" vs " if f["is_home"] else " at ") + f["away"],
                    "kickoff": f["kickoff"],
                    "difficulty": f["difficulty"],
                    "difficulty_source": f["difficulty_source"],
                    "match_id": f["match_id"],
                }
                for f in sorted(team_fixtures, key=lambda item: (item["gw"], item["kickoff"]))
            ],
        })

    return {
        "season": "2026/27",
        "data_gameweek": latest_gw,
        "next_gameweek": target_gameweek,
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "players": profiles,
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameweek", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("live_data.json"))
    parser.add_argument("--predictions", type=Path, default=Path("predictions.json"))
    args = parser.parse_args()

    if not 2 <= args.gameweek <= 38:
        raise SystemExit("--gameweek must be between 2 and 38")

    requested_latest_gw = args.gameweek - 1
    available_stats = []
    skipped_gameweeks = []
    for gw in range(1, requested_latest_gw + 1):
        try:
            frame = read_csv(gw, "player_gameweek_stats.csv")
        except ValueError as exc:
            if "Empty source:" not in str(exc):
                raise
            skipped_gameweeks.append(gw)
            continue
        frame = frame.assign(gw=gw)
        if frame.empty:
            skipped_gameweeks.append(gw)
            continue
        available_stats.append(frame)

    if not available_stats:
        raise ValueError(
            f"No non-empty player_gameweek_stats.csv files found before GW{args.gameweek}"
        )
    current = pd.concat(available_stats, ignore_index=True)
    latest_gw = int(current["gw"].max())
    if skipped_gameweeks:
        print(
            "Warning: skipping empty gameweek stats for "
            + ", ".join(f"GW{gw}" for gw in skipped_gameweeks)
            + f"; latest usable player data is GW{latest_gw}."
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
    prediction_details_by_fixture = {}
    if args.predictions.exists():
        raw = json.loads(args.predictions.read_text(encoding="utf-8"))
        source_players = raw.get("all_players")
        if not isinstance(source_players, list):
            source_players = [
                {
                    **player,
                    "fixture_home": fixture_prediction.get("home"),
                    "fixture_away": fixture_prediction.get("away"),
                }
                for fixture_prediction in raw.get("predictions", [])
                for player in fixture_prediction.get("players", [])
            ]
        for player in source_players:
            fixture_home = player.get("fixture_home")
            fixture_away = player.get("fixture_away")
            if not fixture_home or not fixture_away:
                continue
            canonical_name = NAME_ALIASES.get(player["name"], player["name"])
            key = (canonical_name, fixture_home, fixture_away)
            prediction_value = player.get("prediction")
            if prediction_value is not None:
                predictions_by_fixture[key] = float(prediction_value)
                prediction_details_by_fixture[key] = player

    rows = []
    for _, row in latest.iterrows():
        player_id = int(row["id"])
        team_code = int(row["team_code"]) if "team_code" in row and pd.notna(row["team_code"]) else None
        # Prefer the current players.csv team when the gameweek snapshot has
        # a missing or stale code. This keeps transferred/newly updated players
        # in the public player pool instead of silently dropping them.
        if team_code not in team_map:
            match = players[players["player_id"] == player_id]
            team_code = int(match.iloc[0]["team_code"]) if not match.empty and pd.notna(match.iloc[0]["team_code"]) else None
        fixture = next_fixture.get(team_code)

        name = str(row["web_name"])
        canonical_name = NAME_ALIASES.get(name, name)
        prediction = None
        prediction_details = {}
        if fixture:
            key = (canonical_name, fixture["home"], fixture["away"])
            prediction = predictions_by_fixture.get(key)
            prediction_details = prediction_details_by_fixture.get(key, {})
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
            "prediction_std": safe_float(prediction_details.get("prediction_std")),
            "prediction_p10": safe_float(prediction_details.get("prediction_p10")),
            "prediction_p90": safe_float(prediction_details.get("prediction_p90")),
            "prediction_range": safe_float(prediction_details.get("prediction_range")),
            "confidence": prediction_details.get("confidence"),
            "expected_minutes": safe_float(prediction_details.get("expected_minutes")),
            "start_probability": safe_float(prediction_details.get("start_probability")),
            "sixty_probability": safe_float(prediction_details.get("sixty_probability")),
            "fixture_difficulty": prediction_details.get("fixture_difficulty"),
            "fixture_context": prediction_details.get("fixture_context"),
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

    historical = load_completed_seasons()
    profile_payload = build_player_profiles(
        current,
        players,
        teams,
        args.gameweek,
        prediction_details_by_fixture,
        historical,
    )
    profile_output = args.output.with_name("player_profiles.json")
    profile_output.write_text(
        json.dumps(profile_payload, ensure_ascii=False, indent=2) + "\n",
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
