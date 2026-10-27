# ScoutXI model, data and website improvement roadmap

Last reviewed: 24 October 2026

This roadmap separates facts already present in the repository from improvements that should be implemented and measured. The guiding rule is: **new data is only useful when its provenance is clear, it is available before the prediction deadline, and an out-of-time backtest shows that it improves predictions.**

## Current pipeline audit

The model already uses a substantial feature set: rolling FPL points and minutes; starts/start rates; expected goals and expected assists; ICT, influence, creativity, threat, BPS and bonus rates; historical player and club priors; home/away context; team and opponent xG/xGA context; fixture-environment features; Elo/strength features; and a separate expected-minutes model. The Random Forest predictions also expose tree spread, P10/P90-like values, and expected-minutes/start estimates.

The historical training source is Vaastav's public Fantasy Premier League dataset, with seasons currently listed from 2019–20 through 2025–26 in `model/historical_intelligence.py`. Current-week player data is loaded from the public FPL-Core-Insights GitHub feed in `model/update_predictions.py`. Official FPL endpoints are already used to evaluate realised gameweek points.

Two accuracy limitations were identified and addressed in the linked code change:
1. The model generates an `all_players` prediction pool, but prediction history previously saved only the top three players per fixture. New snapshots now preserve the full eligible-player pool when supplied. Existing historical records remain top-three-only; they cannot be recreated after the fact.
2. The workflow's no-completed-gameweek fallback was malformed JSON. The fallback now includes the gameweek and emits valid JSON.

The accuracy report now records the snapshot scope, expected player count and name-match rate. When older snapshots are evaluated, they are explicitly marked as top-three-only rather than misrepresented as full-pool accuracy.

## Data-source catalogue and safe-use notes

### Priority 1 — Official Fantasy Premier League API
- Entry point: https://fantasy.premierleague.com/api/bootstrap-static/
- Fixture feed: https://fantasy.premierleague.com/api/fixtures/
- Player history: https://fantasy.premierleague.com/api/element-summary/{player_id}/
- Live gameweek results: https://fantasy.premierleague.com/api/event/{event_id}/live/
- Set-piece notes endpoint is documented by community-maintained API reference material at https://oas-official.whatthefpl.com/api.
- Useful fields include official player IDs, club/position, current price, ownership, form, availability flags, season totals, FPL xG/xA where supplied, fixture schedule and realised FPL points. Player history supports per-gameweek minutes, points, bonus/BPS, ICT, xG/xA, value and home/away.
- The API is public but not formally documented or guaranteed stable by the Premier League; schema validation, request timeouts, caching, a descriptive User-Agent, and graceful failure are essential. Use official player IDs for joins wherever possible instead of relying only on player names.

### Priority 2 — Existing historical FPL archive
- Repository: https://github.com/vaastav/Fantasy-Premier-League
- Already used for historical priors and model training. Keep it as a cacheable historical source, validate the expected columns, record the data retrieval date, and review the repository's current licence and data redistribution conditions before republishing derived/raw datasets.
- The code should fail clearly if a season is empty or a required feature disappears. Avoid quietly turning a broad missing-data failure into zero-valued features.

### Priority 3 — Existing current-season data feed
- Repository/feed: https://github.com/olbauday/FPL-Core-Insights
- Already used in the automated prediction pipeline for gameweek player and fixture data. Before expanding reliance, verify its current maintenance status, data dictionary, update cadence and licence. Add source timestamps and feed-health checks; keep official FPL endpoints as the reference source for IDs, gameweek status and realised points.
- Do not assume a feed column is available in every gameweek. Optional features need explicit missingness indicators or a safe fallback that is visible in diagnostics.

### Priority 4 — Official Premier League stats and external xG sources
- Premier League methodology overview: https://www.premierleague.com/en/stats/clarification
- The Premier League states that its performance data is collected and analysed by Opta/Stats Perform and describes its post-match quality-control process. Public website tables are useful for validation, but automated extraction should be reviewed against applicable usage terms and access rules.
- Understat (https://understat.com/) and independent sources such as https://www.xgstat.com/ can be useful for cross-checking xG/xA trends. Their definitions, coverage and player matching may differ from FPL's numbers. Do not blend competing xG definitions into training without documenting the source, checking the licence/terms, and backtesting them separately. Prefer a single consistent xG provider for each trained feature family.

### Research-only / do not add as a live dependency without review
- StatsBomb Open Data: https://github.com/statsbomb/open-data. Useful event-level historical research in selected competitions, not a drop-in live feed for the current Premier League season. The project requests source attribution when publishing analysis based on its data. Check its current terms and competition coverage before use.
- Football-Data.co.uk: https://www.football-data.co.uk/contact.php. Its published fair-use policy limits the free CSV data to personal/academic interest and light manual or automated use, and says free files are not intended for commercial or heavily automated AI training. Do **not** automate its feed into ScoutXI's model without explicit permission or a suitable subscription.
- Injury news, predicted line-ups, press conferences, odds, referee details and weather should be considered only after the model's data lineage and backtesting are reliable. Free availability does not automatically mean permission for automated collection, redistribution, or model training.

## Step-by-step implementation plan

### Phase 0 — Make the pipeline observable and reliable
1. Let the current model backtest and the queued data-refresh workflow finish; inspect the first failure rather than assuming that a successful Pages deployment proves the model refreshed.
2. Confirm `prediction_accuracy_history.json` is created on the first run and that subsequent updates preserve past gameweeks.
3. Confirm new history records report `prediction_scope: full_player_pool` and that `players_expected` is close to the current `all_players` count. Existing records cannot be backfilled if the old full-pool predictions were not saved.
4. Run the targeted history tests and validate the generated prediction, profile, fixture-difficulty and live-data JSON before committing any update.
5. Add a visible `data_updated_at`, data-source status and target gameweek/deadline to generated datasets. If a critical feed is stale or malformed, fail closed instead of silently publishing stale predictions.

### Phase 1 — Establish a trustworthy accuracy baseline
1. Collect frozen predictions for every eligible player before each deadline; never overwrite that snapshot after results arrive.
2. Evaluate only gameweeks marked finished by the official FPL feed. Store the target gameweek and data-source/version metadata with each evaluation.
3. Improve player matching by using official FPL player IDs in snapshots. Where old snapshots only contain names, normalize names and use team/fixture context as a secondary match; report unmatched and ambiguous names explicitly.
4. Report MAE, RMSE, signed error/bias, player match rate and sample size overall, by position, minutes band, confidence band, price band and starting probability.
5. Keep top-three fixture predictions as a separate product-specific measure, but do not treat them as the model's full-player accuracy.
6. Compare ScoutXI against simple baselines (previous-three-gameweek points and a position/availability-aware baseline) on identical out-of-time samples.
7. Wait for enough samples before adjusting features; a handful of players or a single gameweek is not a sound basis for changing model weights.

### Phase 2 — Leakage-safe historical testing
1. Use rolling-origin/walk-forward testing: for each historical gameweek, train using only data available before that deadline, then predict the held-out gameweek.
2. Keep all feature transformations and imputations inside each training fold. Audit that rolling features are shifted and that season/club history never uses future observations.
3. Evaluate across multiple seasons and separate early-season, mid-season and late-season conditions.
4. Save a machine-readable comparison of the current model, simple baselines and each experimental change.
5. Only promote a change if it improves out-of-time MAE/RMSE or another predeclared metric, is not materially worse for a key position, and does not break pipeline or calibration checks.

### Phase 3 — Improve high-value football features one at a time
1. **Expected minutes and starts:** validate predicted minutes and start probability against actual starts/minutes. Separate rotation risk, injury/suspension status, recent manager usage and returning-from-injury players. This is likely to be high value because a good points-per-90 player can still score zero if unused.
2. **Availability freshness:** ingest official FPL availability/status fields and time-stamp the latest observation. Add an explicit uncertainty flag when a player's news is stale or status is doubtful; avoid hard-coded injury assumptions.
3. **Set pieces and penalties:** use the FPL set-piece notes where available; track penalties, corners and free kicks as roles with an observation date. Do not assume a role is permanent after transfers or line-up changes.
4. **Underlying attacking form:** test rolling xG, xA and xGI per 90, shot/shot-on-target rates if a verified legal source is available, and distinguish total volume from per-90 rate. Shrink tiny-minute samples toward a player's longer-term prior.
5. **Team and opponent context:** test recent team xG for/against, opponent defence by venue, home/away scoring rates, and fixture congestion/rest days. Avoid duplicating near-identical features without checking feature importance and out-of-time benefit.
6. **Player role and club moves:** detect transfers, role changes and changes in set-piece duty; blend career priors with current-club priors while shrinking small club samples.
7. **Price and ownership:** use price, ownership and transfer trend primarily for value/differential and squad-building products; only include them in point predictions if tests show they add predictive information rather than just popularity.
8. Add any new feature in isolation, document its source and missing-data behaviour, then rerun the same frozen backtest.

### Phase 4 — Calibrate predictions and explain uncertainty honestly
1. The current P10/P90 values are derived from tree-to-tree Random Forest spread; treat these as a model-spread range, not a calibrated 80% predictive interval.
2. Measure empirical coverage of the published interval on held-out gameweeks, overall and by position/minutes band.
3. Calibrate intervals using out-of-time residuals or a suitable conformal/quantile method. Do not train the calibration on the same gameweek being evaluated.
4. Compare confidence bands to actual error rates. If high-confidence players are not more accurate, change or relabel the confidence calculation.
5. Explain on player pages which inputs are descriptive context and which effects have been validated. Avoid claiming a causal breakdown when the model does not estimate causal contributions.

### Phase 5 — Website trust and decision support
1. Add an accuracy dashboard with gameweek-by-gameweek trends, sample sizes, match rate, baseline comparison and a clear last-updated time.
2. Separate live model predictions, realised accuracy, projected/estimated metrics and unavailable data visually.
3. Improve player pages with minutes/start probability, recent underlying stats, next fixtures, model range and short caveats for low-sample or stale-data cases.
4. Keep team and fixture difficulty labels honest: clearly distinguish official FPL FDR from ScoutXI's own model-estimated difficulty.
5. Add a compact data-source/methodology page with the season, snapshot time, metrics and known limitations.

### Phase 6 — Differentiate the Team Builder
1. Use full-pool predictions and current price/position/team data as its live input.
2. Enforce official squad constraints (15 players, budget, positions and maximum club quota) and keep rules configurable if FPL changes them.
3. Optimise the next 1, 3 and 5 gameweeks, with explicit weight controls for expected points, minutes confidence, fixture difficulty and budget.
4. Provide transparent trade-offs: expected points gained, money remaining, fixtures, minutes risk and predicted range; do not label a squad "optimal" without stating the objective.
5. Add captain/vice-captain suggestions and bench order only after the base squad constraints and scoring are tested.
6. Test recommendation quality against simple alternatives and keep the user's decisions editable; a strong optimiser should explain why a transfer is recommended.

### Phase 7 — Operations and maintenance
1. Cache public datasets where possible and avoid repeated full history downloads on every scheduled run.
2. Record source URL, retrieval time, season/gameweek, row count, schema version and validation result for each dataset.
3. Add retries with backoff for transient failures, request timeouts, and a maximum request volume.
4. Keep source licences/terms in a source register and revisit them before expanding collection or monetisation.
5. Run fast schema/history unit tests first, then expensive backtests; upload backtest artifacts so results are inspectable.
6. Maintain a small release checklist: tests pass, data refresh passes, prediction coverage is healthy, accuracy snapshot is preserved, Pages deployment passes, and the public site loads.

## Immediate next action

Once the current workflows finish, inspect the data-refresh logs. Then confirm that the first accuracy record is honest about the old top-three-only Gameweek 8 snapshot, and that the next frozen gameweek snapshot captures the full eligible-player pool. Only after those checks should model features be changed.


## Step 1 — free-source audit: team news and cup appearances

Reviewed 27 October 2026. This is a source-selection decision, not a claim that either data feed is integrated.

### Confirmed team news

- **Primary, already integrated:** official FPL bootstrap fields (`status`, `news`, and chance-of-playing fields). Treat these as FPL-published availability signals, not independent confirmation.
- **Candidate for manual cross-checking:** WassupFPL's public press-conference page describes a free account and club-by-club availability digests. Its public page does not establish a supported, public API or reuse licence for automated ingestion. Do not scrape the site or ingest its content into the public snapshot until an authorised API/export and terms for reuse are confirmed.
- **Preferred source if automation is later authorised:** official club and manager press-conference pages, retaining source URL, publication time, player, exact short evidence/quote where permitted, verdict, and confidence. A manually curated link is safer than treating a headline or model-generated summary as a verified fact.

### Domestic and European cup appearances

- The current official FPL API provides Premier League fantasy-season minutes, not a complete cross-competition player-minute history. It cannot by itself confirm cup rotation.
- football-data.org's free plan documents fixtures/schedules and delayed scores; its pricing page places line-ups/substitutions outside the basic free plan. It is therefore not sufficient on its own for reliable cup-minute ingestion.
- A third-party lineup API with a free quota is only a *candidate* until its current competition coverage, cup line-up/substitution fields, retention/reuse terms, and quota are verified directly. Do not make it a production dependency based only on a third-party comparison page.

### Decision and acceptance criteria

1. Keep the current snapshot flags explicit: `confirmed_team_news: false` and `cup_appearances: false`. Never infer these values from the presence of FPL injury text or fixture lists.
2. Do not add a new API key, paid plan, brittle scraping job, or unverified data source merely to make coverage look complete.
3. Before implementation, verify the provider's own documentation and terms; test a representative current-season player and cup match; confirm the response contains player-level evidence and timestamps; and document attribution/retention limits.
4. For team news, mark information as pre-deadline only when its publication timestamp precedes the prediction cutoff. Preserve the original source link and distinguish confirmed, doubtful, and unknown.
5. For cup rotation, store appearances, starts, minutes and competition separately from Premier League minutes; use null for missing data, not zero.
6. Any new feature must pass schema/coverage checks and an out-of-time backtest against the existing baseline before it can influence predictions.

**Source references checked:**
- WassupFPL press-conference feed: https://www.wassupfpl.com/press-conference
- football-data.org free coverage: https://www.football-data.org/coverage
- football-data.org pricing and free-tier limits: https://www.football-data.org/pricing
- football-data.org API documentation: https://www.football-data.org/documentation/api
