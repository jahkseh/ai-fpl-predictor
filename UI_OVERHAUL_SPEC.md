# AI FPL Predictor — UI Overhaul Specification

## Purpose

Turn the existing static GitHub Pages application into a focused FPL decision workspace. The user should be able to answer three questions within a few seconds:

1. Who are the strongest projected picks for the next Gameweek?
2. Which player, transfer or captaincy decision could improve my team?
3. Why does the model recommend that decision, and how uncertain is the projection?

This is an incremental frontend refresh, not a framework migration or a rewrite of the prediction model.

## Current implementation constraints

The site is currently a static HTML/CSS/JavaScript app with data published as JSON. Existing entry points include:

- `index.html` — dashboard, predictions, player discovery and supporting sections.
- `player.html` — player profile and historical/current performance details.
- `team.html` — club/team profile, squad picks and upcoming fixture context.
- `predictions.json`, `live_data.json`, `player_profiles.json`, `differentials.json` and prediction-history/audit files — published prediction and supporting data.
- `model/` and `.github/workflows/` — prediction/data generation and validation.

Preserve these files, their current data contracts and the existing data-refresh workflow. Before renaming or removing any current element, inspect its readers, anchors and test coverage. Do not fabricate unavailable values or present missing data as zero. Use the real values already published in the repository; introduce graceful empty states when a field is absent.

## 1. Visual direction

**Style:** contemporary sports analytics — calm, dense enough for comparison, but not dashboard clutter. Make the data the visual focus; avoid excessive decoration, glowing borders, oversized hero content and animations that do not communicate information.

### Design tokens

Use the same tokens on every page, ideally in one shared CSS file or a clearly mirrored token block until styles can safely be extracted.

| Token | Dark mode | Light mode |
|---|---|---|
| Page background | `#0B1220` | `#F5F7FA` |
| Card surface | `#111827` | `#FFFFFF` |
| Raised / subtle surface | `#162033` | `#EEF2F6` |
| Border | `#243047` | `#DCE3EA` |
| Primary text | `#E5E7EB` | `#172033` |
| Secondary text | `#94A3B8` | `#596579` |
| Primary accent | `#27D17F` | `#159957` |
| Warning | muted amber | dark amber |
| Danger | accessible muted red | dark red |

- Typography: use Inter if it is already loaded or can be added cleanly; otherwise use a system sans-serif stack. Use tabular numerals for points, prices, percentages and rank values.
- Use a 4/8-point spacing scale; cards use 16–20px padding on desktop and 12–16px on mobile.
- Card radius: 12–16px. Main content width: approximately 1200px. Keep body copy comfortable at 14–16px.
- Reserve accent green for primary actions, positive model deltas and selected states. Do not make every border, hover and heading green.
- Team colours can identify clubs, but must not replace readable club names or accessible text labels.
- Respect `prefers-reduced-motion`. Hover effects should be subtle and must not move content in ways that disrupt scanning.
- Keep dark and light themes fully readable; persist the user's choice and provide a reliable fallback when storage is unavailable.

## 2. Layout and navigation

### Shared application shell

Create a consistent shell for dashboard, player, team and planner pages:

- Left: compact wordmark.
- Main links: **Dashboard**, **Predictions**, **Players**, **Team Planner**, **Teams**, **Model Performance**.
- Right: active Gameweek indicator, data-updated timestamp where available, and a compact theme toggle.
- Clearly style the active page and provide visible keyboard focus.
- At narrow widths, collapse the navigation into a menu or a deliberate horizontally scrollable nav. Avoid the current pattern of turning every link into a large grid button.

### Dashboard hierarchy

1. **Compact page header:** “Gameweek [n] outlook”, deadline context when available, data freshness label and a brief model status note. Replace a tall promotional hero with a concise, task-oriented header.
2. **At-a-glance metric row:** model-projected team/pick total where valid, top projected player, number of high-confidence picks, and data freshness. Only show metrics that have a defensible source; do not invent overall accuracy or confidence.
3. **Top AI picks:** a ranked compact table/list with player, club, position, fixture, projected points, expected minutes/status if available, ownership/differential context and a short reason.
4. **Decision cards:** top captaincy candidate, best value/differential and a transfer/planning prompt. Each card explains the reason and uncertainty, not just a large number.
5. **Fixture outlook:** concise next-Gameweek fixture difficulty view, followed by the next 3–5 Gameweeks where data exists.
6. **Prediction history/performance:** compact actual-vs-predicted summary with a link to detailed methodology and historical performance.
7. Keep secondary educational/feature content lower on the page so it does not compete with current decisions.

On desktop, use a 12-column grid: primary prediction/ranking area takes about 7–8 columns; contextual insights take 4–5. Stack in reading order on mobile.

## 3. Data visualization and prediction language

### Expected-points presentation

- Show **projected points** as the main value, e.g. `5.8 pts`, with the Gameweek and fixture immediately nearby.
- When the model provides a defensible uncertainty interval, show a low-to-high band (P10–P90 or the model's actual equivalent) as a thin range bar. Label the interval explicitly. Do not imply that it is a guaranteed outcome.
- Keep model confidence distinct from projected points and from historical prediction accuracy. Never label a pick “90% likely” unless the model is calibrated to support that probability.
- Explain confidence using plain labels (e.g. High / Medium / Low) only when a documented rule or calibrated field supports them. Add a tooltip or “Why?” details panel with the top contributing factors.
- Explain projections with available inputs such as expected minutes, recent form, underlying attacking output, team strength, opponent defence, home/away and fixture context. Hide absent inputs; never render `0` as a fallback for unknown history.
- Show the data Gameweek/source freshness in small muted text.

### Charts

Prefer a few useful visualizations over many charts:

- Player points: recent actual points and projected points should be visually differentiated and labelled.
- Player profile: trend line/bar chart plus a compact underlying-metrics table (minutes, starts, xG, xA, points per 90, only where available).
- Team profile: next 3–5 fixtures with opponent and clearly labelled difficulty, plus attacking/defensive summary measures if present.
- Model performance: compare predicted with actual values over completed Gameweeks; include sample size and evaluation method.

Chart requirements: readable axis labels, accessible tooltip content, no reliance on colour alone, responsive sizing, and a text/table alternative for key data. Avoid decorative gauges that imply precision without a defined interpretation.

### Colour conventions

- Green: selected/positive signal or primary action.
- Amber: caution, rotation risk or moderate fixture difficulty.
- Red: adverse signal or high risk; avoid using it for ordinary neutral information.
- Fixture difficulty must have both a number/label and a colour cue.
- Use a separate neutral visual style for the model's uncertainty interval so it is not mistaken for a good/bad signal.

## 4. Player discovery and profile

### Player discovery

Improve the existing player search/filter area rather than creating a separate, disconnected catalogue.

- Search by player name with forgiving case/spacing and immediate feedback.
- Filters: position, club, price range, projected points, expected minutes (when available), ownership, confidence/uncertainty and home/away.
- Sort: projected points, value (points per £m), ownership, price, expected minutes and fixture difficulty.
- Show active filters as removable chips, with **Clear all** always visible once a filter is applied.
- Display result count and a useful empty state with suggestions to remove a filter or check spelling.
- Do not hide the entire player list by default unless the interface makes it obvious how to reveal results; the first visit must not look broken.
- On narrow screens, expose filters in a drawer or expandable panel rather than compressing a long row of controls.

### Player card / table

Prioritise player name, club, position, next opponent, projected points and price. Secondary values should remain subdued. Use real club names consistently and make each card/table row a clear link to the player profile.

### Player profile information order

1. Name, club, position, price, next fixture and projected points.
2. Projection range/confidence explanation and “Why this projection?” factors.
3. Recent performance chart and key underlying metrics.
4. Upcoming fixture strip for 3–5 Gameweeks.
5. Historical ability / current form / team attack / opponent defence / expected minutes detail, clearly distinguishing current-club data from cross-club historical data.
6. Sources, update timestamp and available/missing-data notes.

The profile should tell the user what to do with the information, not merely list statistics.

## 5. Team Planner — key product journey

Add a dedicated `planner.html` page (or equivalent self-contained planner route) to keep squad decisions separate from the overview dashboard. Follow the existing static-page pattern and consume the published JSON. This should become the strongest differentiator: AI recommendations are evaluated against the user's actual squad and budget rather than shown as generic picks.

### Desktop planner layout

- **Left / primary workspace:** squad pitch or formation view, grouped by goalkeeper, defence, midfield and attack; each player card includes name, club, projected points and selected/captain/bench state.
- **Right / planning panel:** bank remaining, squad value if available, transfers used/free transfers if supplied by user input, points hit, projected starting XI total and projected bench contribution.
- **Below the pitch:** a searchable player pool with position, price, club, points projection, minutes/status and fixture filters.
- **Transfer queue:** outgoing player → incoming player, price difference, budget after move and projected impact.
- **Horizon selector:** compare the next Gameweek with a 3- and 5-Gameweek projection where supported.

### Core interactions

1. User selects/adds their squad or enters player IDs/names using the published player data.
2. User clicks a player to select an outgoing slot, then chooses a valid incoming player.
3. The planner validates squad structure and the existing budget rules before applying a move.
4. Show projected-points change, confidence/uncertainty difference, fixture sequence and price delta.
5. Allow users to queue multiple moves, undo a move and reset the draft.
6. Make captain, vice-captain, starting XI and bench selection explicit.
7. Include scenario comparison rather than a single prescriptive answer: “Current squad” vs “Proposed changes”.

Do not claim to submit transfers to the official FPL account unless an authorised integration actually exists. State clearly when squad data is stored only locally. If the user has not connected or entered a squad, provide an obvious setup path rather than empty cards.

### Chip strategy

Create a focused chip panel inside the planner, not a generic callout on every page.

- Explain which future weeks are comparatively attractive for a chip using fixture quality, availability, expected minutes, projected team points and the remaining schedule when those inputs exist.
- Compare **Use now** versus **Save for later** scenarios over the same planning horizon.
- Show assumptions and a short rationale for the recommendation.
- Ask users to confirm their remaining chips and relevant squad context if it is not available.
- Distinguish estimated gain from guaranteed gain; expose uncertainty and avoid false precision.

### Budget and validation

- Maintain a single source of truth for prices, player IDs and position/team restrictions.
- Show validation inline: insufficient funds, duplicate player, invalid squad composition or club limit.
- Do not silently discard a selected player if data reloads or a filter changes.
- Provide undo/reset controls with confirmation when a user could lose multiple planned moves.
- Keep a clear visual distinction between the saved squad and uncommitted draft changes.

## 6. Team profile

Retain the existing club-colour identity but reduce ornamental effects. Order information as:

1. Club identity, next fixture and projected team context.
2. Compact snapshot metrics.
3. Highest projected players with profile links.
4. Next 3–5 fixtures and labelled difficulty.
5. Relevant attacking/defensive statistics and notes about missing inputs.

Use the same header, card, badge, type and responsive tokens as the dashboard and player page.

## 7. Reusable frontend component structure

Keep components as small reusable HTML partials/functions or JavaScript render helpers in the current architecture; no React/Vue rewrite is required.

- `AppShell / SiteHeader` — navigation, active route, theme.
- `PageHeader` — title, Gameweek context, update timestamp.
- `MetricCard` — label, value, delta/source note.
- `PlayerIdentity` — name, club, position, price.
- `ProjectionValue` — points, horizon and optional uncertainty interval.
- `ConfidenceIndicator` — documented confidence label plus explanation.
- `FixtureBadge / FixtureStrip` — opponent, home/away and difficulty.
- `PlayerSearch` + `PlayerFilters` + `PlayerResults` — shared discovery interaction.
- `RecommendationCard` — recommendation, reason, downside and action.
- `PointsTrendChart` / `PredictionVsActualChart` — visual summaries plus textual alternatives.
- `SquadPitch` + `SquadPlayerSlot` — squad positions and player states.
- `TransferQueue` + `TransferImpact` — pending changes and projected deltas.
- `BudgetSummary` — remaining budget and validation.
- `ChipScenarioPanel` — compare use-now and save-later scenarios.
- `EmptyState`, `LoadingState`, `ErrorState` — explicit and consistent states.

For now, reuse the existing data-loading code and page contracts. Only share data helpers after confirming they work on GitHub Pages (relative URLs, no backend assumptions).

## 8. Responsive and accessibility requirements

### Breakpoints

- Wide desktop: 1200px and above — split-grid dashboard and planner.
- Tablet: 800–1199px — narrower two-column layouts, filters can wrap.
- Mobile: 560–799px — mostly single column with compact list rows.
- Small mobile: 559px and below — stacked cards, compact headers, horizontal fixture strips where helpful, and full-width primary actions.

Use fluid typography and CSS grid/flex rather than fixed widths. No horizontal page overflow at 320px. Tables may scroll inside their own labelled container, but the whole page must not overflow.

### Accessibility

- Every action works with keyboard only; visible `:focus-visible` styles.
- Inputs have visible labels; icons/buttons have accessible names.
- Use semantic headings, buttons for actions and anchors for navigation.
- Meet WCAG AA contrast as far as practical; do not use colour as the only indicator.
- Respect reduced-motion settings.
- Charts have summaries and accessible data alternatives.
- Status, error and successful actions are announced appropriately to assistive technologies.
- Touch targets should generally be at least 40–44px high on mobile.

## 9. Technical approach and safety rails

- Keep the existing static GitHub Pages deployment and current data-refresh workflow.
- No new paid APIs or build system are required for the first pass.
- Preserve JSON schemas; do not edit prediction values as part of a visual-only change.
- Build the planner against the fields that actually exist. Add adapter functions for differences between live data, profile data and prediction data rather than scattering field fallbacks across UI code.
- Centralise safe number formatting and missing-value handling. Missing is `—` or “Not available”, never a fabricated zero.
- Keep UI rendering separate from prediction logic. Frontend redesign must not silently change model outputs.
- Avoid embedding large player JSON payloads into multiple new HTML pages; fetch the existing files on demand and handle fetch failures gracefully.
- Use relative file links and asset paths so pages continue working under the GitHub Pages project path.
- Preserve a usable static fallback where the data files already support one.
- Prefer progressive enhancement: core page/navigation remains understandable if a chart or optional JSON refresh fails.

## 10. Delivery plan

### Phase 1 — Consistent visual foundation
- Standardise tokens, typography, card spacing, page headers, navigation and light/dark mode across all three current pages.
- Reduce oversized hero and decorative effects; establish a dashboard-style first viewport.
- Add clear loading, stale-data and empty states.
- Regression-check existing anchors and the prediction/data refresh workflow.

### Phase 2 — Dashboard and player discovery
- Promote next-Gameweek projections and explainable recommendations.
- Fix and test search/filter/sort and empty-state behaviour.
- Standardise player identity, projected points, fixtures and links.

### Phase 3 — Player/team analytical views
- Reorder profile content around decisions.
- Add/standardise readable trends and fixture strips using available values.
- Clearly distinguish unavailable values from real zeroes and historical club rates from current-club measures.

### Phase 4 — AI Team Planner
- Implement squad selection, budget validation, transfer queue, starting XI/bench and projected change.
- Add multi-Gameweek horizon comparisons.
- Implement chip scenario comparison once the required inputs are available.
- Persist only user-created draft data locally unless a real account-backed service is added.

### Phase 5 — Quality and validation
- Test desktop, tablet and small-screen layouts.
- Test direct page loads and relative links on GitHub Pages.
- Test light/dark mode, keyboard-only paths, empty JSON, failed JSON fetches and missing fields.
- Run the repository's existing UI/model checks and compare published prediction data before and after the UI-only work.
- Check for JavaScript console errors, horizontal overflow and inaccessible controls.

## 11. Acceptance criteria

The overhaul is ready when:

- A user can identify top projections and the upcoming Gameweek without scrolling past a large promotional section.
- Every recommendation includes a clear reason and does not overstate certainty.
- Player search works by name and combinations of filters; no-result states explain recovery.
- Projected points, prices, fixtures and club names are consistent on dashboard, player and team pages.
- The planner makes budget and transfer impact visible before the user commits a proposed move.
- Chip advice compares scenarios and explicitly identifies assumptions.
- Missing data never masquerades as zero, and no unsupported “accuracy” or probability claim is displayed.
- The site remains static-host compatible, preserves the prediction pipeline, works at 320px width and has keyboard-visible focus states.
- UI changes do not alter the underlying published prediction JSON values unless a separately reviewed model-data change intentionally does so.

## Recommended implementation order

Start with **Phase 1 and Phase 2**. They deliver the most visible improvement while protecting the working data pipeline. Build the planner as a separate page after the shared components and data adapters are stable, so new squad state cannot accidentally break the dashboard or prediction-generation workflow.
