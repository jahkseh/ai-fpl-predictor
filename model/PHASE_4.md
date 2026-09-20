# Phase 4 — Team and Opponent Intelligence

Phase 4 adds a fixture environment layer to the predictor.

## Features

The model now considers leakage-safe historical team rates for the player's team and opponent:

- team xG per game
- team goals per game
- team xG conceded per game
- team goals conceded per game
- opponent xG per game
- opponent goals per game
- opponent xG conceded per game
- opponent goals conceded per game
- attack mismatch
- defensive mismatch
- combined fixture xG environment

Team features are calculated only from completed gameweeks. For a target gameweek, the current-season team features use only earlier current-season matches.

## Why this matters

A player's historical attacking ability is not enough. The same player can have very different expected output depending on:

1. his own ability and form;
2. his team's attacking environment;
3. the opponent's defensive environment;
4. the opponent's attacking threat;
5. the home/away fixture.

The production prediction JSON exposes these fixture signals so the website can display and audit them.

## Backtesting

The walk-forward experiment now compares:

- baseline: recent player form;
- phase3: recent form + historical player intelligence;
- phase4: Phase 3 + team/opponent intelligence.

The comparison uses identical unseen 2025/26 gameweeks.

## Data safety

No target-gameweek goals, xG, FPL points or team performance are used when constructing the target's historical team features.
