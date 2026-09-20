# Phase 1–2 implementation notes

Phase 1 validation:
- Historical GW33–38 holdout remains a regression guard.
- A previous-three-gameweek points baseline is recorded alongside model metrics.
- Existing leakage checks and the GW6 golden prediction regression remain active.

Phase 2 prediction engine:
- Random Forest tree predictions provide a model-spread estimate.
- Prediction records include P10/P90 uncertainty ranges and confidence bands.
- Fixture context includes opponent, home/away status, team/opponent strength and Elo.
- Legacy prediction fields remain unchanged for compatibility with existing consumers.
