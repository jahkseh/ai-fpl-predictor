# ScoutXI Intelligence Pipeline — Phase 1

## Goal
Create a reliable, timestamped foundation for daily ScoutXI intelligence without changing the existing point-prediction model.

## Current implementation
- `model/build_intelligence_snapshot.py` collects player availability/news fields, price, ownership, form and minutes from the public FPL bootstrap endpoint, plus fixtures and kickoff data from the FPL fixtures endpoint.
- `intelligence_snapshot.json` is a machine-readable snapshot with collection timestamp, source attribution, schema version, coverage counts and explicit flags for sources/features not integrated yet.
- `.github/workflows/daily-intelligence.yml` runs daily at 08:00 UTC (09:00 in UK summer time), supports manual dispatch, validates the snapshot and commits changes only when content differs.
- The existing prediction update workflow is left untouched.

## Important limitations
The first phase does not yet ingest cup match minutes, independent injury/press-conference reporting, xG/xA feeds or betting odds. FPL's own `news` and chance-of-playing fields are captured as source data, not treated as independently verified reports. No new data is fed into the prediction model in this phase.

## Next phases
1. Validate this workflow against live API responses and review the produced snapshot.
2. Add a source adapter for match/cup appearances with reliable IDs and timestamps.
3. Add reputable team-news sources with URL, publication time, player/team IDs and a confidence/status label; preserve original source text separately from any normalized summary.
4. Assess legal terms, coverage, refresh rate and cost for xG/xA and odds providers before integration. Store market snapshots with bookmaker/source, market type, captured time and decimal odds; never infer player xG directly from odds.
5. Create a daily change detector comparing snapshots and existing predictions, with thresholds and reasons for material changes.
6. Publish an Intelligence page and hidden-gem shortlist only when the underlying metrics are available. Rank value using predicted points, price, minutes security, ownership and upcoming fixtures, and clearly label uncertainty.
7. Evaluate each new model feature via leakage-safe walk-forward backtests before it can affect official predictions.

## Safety and reliability rules
- Missing values remain null; never turn missing data into zero.
- All fields must be timestamped and source-attributed.
- Validate schema and player/team IDs before publication.
- Do not store API secrets in public JSON or frontend code.
- Keep data collection separate from model features until historical validation shows measurable benefit.
