# ScoutXI external intelligence — secure feed rollout

## Status and safety
This pipeline is separate from the prediction model. It never modifies model inputs or published points. Missing values stay unavailable rather than being converted to zero.

The collector is fail-closed:
- A provider is not called unless its API key is stored as a GitHub Actions secret AND its matching repository Actions variable is explicitly set to the string `true`.
- API keys are passed only to the server-side GitHub Actions process. They are never written to the snapshot, frontend, or logs.
- No payment details are requested or stored by ScoutXI.
- The UK Odds API collector has a hard 240-request monthly cap, below the provider's published 300-request free quota. It requests core fixture markets only.
- Provider failures do not trigger paid fallbacks.
- Odds movement is a match-level signal for investigation. It must never be described as proof that a named player is injured.
- The model remains unchanged pending leakage-safe walk-forward backtesting.

## Provider coverage
### PitchAPI
The documented API exposes a league catalogue, played league fixtures, and per-player advanced match analytics including `minutes_played`. The collector checks recent FA Cup, EFL/Carabao Cup and UEFA competition matches and stores player/team provider IDs, competition, match date and minutes. It deliberately does not guess FPL player IDs from short names.

### UK Odds API
The collector requests the next two days of Premier League core fixture odds and records timestamped snapshots. The monthly request counter is persisted in `odds_usage.json` and stops at 240. Player props and advanced markets are not requested.

## Important publication gate
A free API plan is not itself a licence to republish provider data. Public-use approval variables must remain unset until the provider confirms in writing that ScoutXI may display the data and derived signals on its public website under the free plan. Until then, each adapter remains disabled even if a secret is present.

## GitHub setup (only after provider approval)
Add keys under **Settings → Secrets and variables → Actions → Secrets**:
- `PITCHAPI_API_KEY`
- `UK_ODDS_API_KEY`

Only after written public-use permission, add the corresponding Actions variables:
- `PITCHAPI_PUBLIC_USE_APPROVED=true`
- `UK_ODDS_PUBLIC_USE_APPROVED=true`

Do not add keys to repository variables, source files, website code, commits, issues, screenshots or chat. If a provider asks for card details, a paid subscription, or a billing method, do not proceed; this integration does not require it and has no paid fallback.

## Not yet integrated
Independent manager press-conference reporting is not enabled. The existing official FPL status/news fields remain the availability baseline until a source with clear public-display permission and reliable publication timestamps is approved. No BBC RSS or bookmaker-page scraping is used.
