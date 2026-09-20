"""Leakage-safe historical player intelligence for FPL modelling.

The module builds player-level priors from multiple completed seasons and the
current season's already-played gameweeks. A row never sees its own gameweek
when calculating a historical feature.

External source: Vaastav Fantasy-Premier-League public gameweek CSVs.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

SEASON_URLS = {
    "2022-23": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2022-23/gws/merged_gw.csv",
    "2023-24": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-24/gws/merged_gw.csv",
    "2024-25": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/gws/merged_gw.csv",
    "2025-26": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2025-26/gws/merged_gw.csv",
}

RATE_FEATURES = {
    "total_points": "historical_points_per_90",
    "expected_goals": "historical_xg_per_90",
    "expected_assists": "historical_xa_per_90",
    "goals_scored": "historical_goals_per_90",
    "assists": "historical_assists_per_90",
}

CLUB_RATE_FEATURES = {
    "total_points": "club_points_per_90",
    "expected_goals": "club_xg_per_90",
    "expected_assists": "club_xa_per_90",
}

HOME_AWAY_FEATURES = {
    "total_points": "split_points_per_90",
    "expected_goals": "split_xg_per_90",
    "expected_assists": "split_xa_per_90",
}


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return text


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.replace(0, np.nan) * 90).replace([np.inf, -np.inf], np.nan)


def _prepare(df: pd.DataFrame, season: str) -> pd.DataFrame:
    out = df.copy()
    out["season"] = season
    out["GW"] = pd.to_numeric(out["GW"], errors="coerce")
    out = out[out["GW"].between(1, 38)].copy()

    if "name" not in out.columns:
        if {"first_name", "second_name"} <= set(out.columns):
            out["name"] = out["first_name"].astype(str) + " " + out["second_name"].astype(str)
        else:
            raise ValueError(f"{season}: no player name column")

    if "team" not in out.columns:
        out["team"] = out.get("team_name", "unknown")

    out["player_key"] = out["name"].map(normalize_name)
    out["club_key"] = out["team"].map(normalize_name)

    numeric_defaults = [
        "minutes", "starts", "total_points", "goals_scored", "assists",
        "expected_goals", "expected_assists",
    ]
    for col in numeric_defaults:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    out["was_home"] = pd.to_numeric(out.get("was_home", 0), errors="coerce").fillna(0).astype(int)
    out = out.sort_values(["player_key", "season", "GW"]).reset_index(drop=True)
    return out


def load_completed_seasons(seasons: dict[str, str] | None = None) -> pd.DataFrame:
    frames = []
    for season, url in (seasons or SEASON_URLS).items():
        frame = pd.read_csv(url)
        if frame.empty:
            raise ValueError(f"Historical source returned no rows: {season}")
        frames.append(_prepare(frame, season))
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(["season", "element", "GW"], keep="last")
    return combined.sort_values(["player_key", "season", "GW"]).reset_index(drop=True)


def add_historical_priors(df: pd.DataFrame) -> pd.DataFrame:
    """Add prior career, club, season and home/away rates without leakage."""
    out = df.sort_values(["player_key", "season", "GW"]).copy()
    out["_minutes_cum"] = out.groupby("player_key")["minutes"].cumsum().shift(1)
    out["_points_cum"] = out.groupby("player_key")["total_points"].cumsum().shift(1)
    out["_xg_cum"] = out.groupby("player_key")["expected_goals"].cumsum().shift(1)
    out["_xa_cum"] = out.groupby("player_key")["expected_assists"].cumsum().shift(1)
    out["_goals_cum"] = out.groupby("player_key")["goals_scored"].cumsum().shift(1)
    out["_assists_cum"] = out.groupby("player_key")["assists"].cumsum().shift(1)
    out["_starts_cum"] = out.groupby("player_key")["starts"].cumsum().shift(1)

    for source, name in RATE_FEATURES.items():
        out[name] = _safe_rate(out[f"_{source.replace('total_points','points').replace('expected_goals','xg').replace('expected_assists','xa').replace('goals_scored','goals').replace('assists','assists')}_cum"] if False else pd.Series(dtype=float), pd.Series(dtype=float))

    out["historical_points_per_90"] = _safe_rate(out["_points_cum"], out["_minutes_cum"])
    out["historical_xg_per_90"] = _safe_rate(out["_xg_cum"], out["_minutes_cum"])
    out["historical_xa_per_90"] = _safe_rate(out["_xa_cum"], out["_minutes_cum"])
    out["historical_goals_per_90"] = _safe_rate(out["_goals_cum"], out["_minutes_cum"])
    out["historical_assists_per_90"] = _safe_rate(out["_assists_cum"], out["_minutes_cum"])
    out["historical_start_rate"] = out["_starts_cum"] / out.groupby("player_key")["GW"].cumcount().replace(0, np.nan)

    # Club-specific history: retain information when a player changes club.
    for metric, prefix in CLUB_RATE_FEATURES.items():
        value_col = {"total_points": "_club_points", "expected_goals": "_club_xg", "expected_assists": "_club_xa"}[metric]
        group = out.groupby(["player_key", "club_key"])[metric]
        cumulative = group.cumsum().shift(1)
        mins = out.groupby(["player_key", "club_key"])["minutes"].cumsum().shift(1)
        out[value_col] = cumulative
        out[prefix] = _safe_rate(cumulative, mins)

    club_starts = out.groupby(["player_key", "club_key"])["starts"].cumsum().shift(1)
    club_rows = out.groupby(["player_key", "club_key"]).cumcount()
    out["club_start_rate"] = club_starts / club_rows.replace(0, np.nan)

    # Previous completed season. This captures a player's established level
    # while allowing the current season to deviate from it.
    season_totals = (
        out.groupby(["player_key", "season"], as_index=False)
        .agg(
            season_minutes=("minutes", "sum"),
            season_points=("total_points", "sum"),
            season_xg=("expected_goals", "sum"),
            season_xa=("expected_assists", "sum"),
        )
        .sort_values(["player_key", "season"])
    )
    season_totals["previous_season_points_per_90"] = _safe_rate(
        season_totals["season_points"].shift(1),
        season_totals["season_minutes"].shift(1),
    )
    season_totals["previous_season_xg_per_90"] = _safe_rate(
        season_totals["season_xg"].shift(1),
        season_totals["season_minutes"].shift(1),
    )
    season_totals["previous_season_xa_per_90"] = _safe_rate(
        season_totals["season_xa"].shift(1),
        season_totals["season_minutes"].shift(1),
    )
    season_totals = season_totals.drop(columns=["season_minutes", "season_points", "season_xg", "season_xa"])
    out = out.merge(season_totals, on=["player_key", "season"], how="left", validate="many_to_one")

    # Home/away split history. The split is only calculated from earlier rows.
    split = out.groupby(["player_key", "was_home"])
    split_points = split["total_points"].cumsum().shift(1)
    split_xg = split["expected_goals"].cumsum().shift(1)
    split_xa = split["expected_assists"].cumsum().shift(1)
    split_minutes = split["minutes"].cumsum().shift(1)
    out["split_points_per_90"] = _safe_rate(split_points, split_minutes)
    out["split_xg_per_90"] = _safe_rate(split_xg, split_minutes)
    out["split_xa_per_90"] = _safe_rate(split_xa, split_minutes)

    drop = [c for c in out.columns if c.startswith("_")]
    return out.drop(columns=drop)


def historical_feature_columns() -> list[str]:
    return [
        "historical_points_per_90",
        "historical_xg_per_90",
        "historical_xa_per_90",
        "historical_goals_per_90",
        "historical_assists_per_90",
        "historical_start_rate",
        "club_points_per_90",
        "club_xg_per_90",
        "club_xa_per_90",
        "club_start_rate",
        "previous_season_points_per_90",
        "previous_season_xg_per_90",
        "previous_season_xa_per_90",
        "split_points_per_90",
        "split_xg_per_90",
        "split_xa_per_90",
    ]


def build_current_history_lookup(historical: pd.DataFrame) -> pd.DataFrame:
    """Return the latest historical priors for each player name."""
    cols = ["player_key"] + historical_feature_columns()
    lookup = historical.sort_values(["player_key", "season", "GW"]).groupby("player_key").tail(1)[cols]
    return lookup.drop_duplicates("player_key").reset_index(drop=True)


def build_player_history_report(historical: pd.DataFrame) -> pd.DataFrame:
    """Compact player/club history table useful for diagnostics and UI."""
    result = (
        historical.groupby(["player_key", "name", "season", "club_key"], as_index=False)
        .agg(
            minutes=("minutes", "sum"),
            points=("total_points", "sum"),
            xg=("expected_goals", "sum"),
            xa=("expected_assists", "sum"),
            starts=("starts", "sum"),
            goals=("goals_scored", "sum"),
            assists=("assists", "sum"),
        )
    )
    result["points_per_90"] = _safe_rate(result["points"], result["minutes"])
    result["xg_per_90"] = _safe_rate(result["xg"], result["minutes"])
    result["xa_per_90"] = _safe_rate(result["xa"], result["minutes"])
    return result.sort_values(["player_key", "season", "club_key"]).reset_index(drop=True)
