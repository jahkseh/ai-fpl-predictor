# Phase 3 — Predictive Intelligence

Phase 3 upgrades the predictor from short-term form only to a leakage-safe player intelligence system.

## Implemented

### Historical player intelligence
- 2022/23, 2023/24, 2024/25 and 2025/26 history.
- Career points, xG, xA, goals and assists per 90.
- Historical start rate.
- Previous completed-season points/xG/xA per 90.
- Player-at-club points/xG/xA per 90 and club start rate.
- Home/away historical splits.
- Player identity is normalized across seasons, so transfers retain history.

A player moving clubs does not lose their historical profile. Current-club fixture and current-form information remain separate.

### Expected minutes
A separate Random Forest estimates:
- probability of starting;
- probability of reaching 60 minutes;
- expected minutes.

These are exposed in prediction JSON and do not use the target gameweek's minutes or points.

### Workload
Recent gameweek match/minute load is available as leakage-safe modelling features. European/cup workload is not fabricated when the underlying source does not provide reliable player appearance data.

### Controlled backtesting
phase3_backtest.py performs a walk-forward comparison on the same unseen 2025/26 gameweeks:
- baseline = existing recent-form feature set;
- phase3 = recent form + historical player intelligence.

The pipeline reports MAE, RMSE and R² per gameweek and averaged across the evaluation window.

## Data leakage rules

1. A player's own target-gameweek statistics are never used as a feature.
2. Historical cumulative features use only rows before the prediction row.
3. Completed-season features only use seasons that ended before the target row.
4. Missing historical experience is represented conservatively rather than inventing performance.
5. Model changes are evaluated on identical unseen gameweeks.

## Next modelling layer

The next controlled experiments should add:
1. historical team attack/defence and xG/xGA;
2. opponent-specific defensive environment;
3. richer fixture difficulty;
4. reliable player appearance data for European/cup workload;
5. a two-stage points model that explicitly converts expected minutes into expected FPL points;
6. model comparison with gradient boosting/XGBoost only after feature experiments establish measurable out-of-sample improvement.
