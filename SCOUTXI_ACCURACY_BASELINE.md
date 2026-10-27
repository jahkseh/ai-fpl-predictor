# ScoutXI accuracy baseline — 27 October 2026

## Current measured result

The published `prediction_accuracy.json` evaluates Gameweek 8 against completed official FPL points.

| Metric | Current value |
|---|---:|
| Players evaluated | 28 |
| Unmatched player records | 2 |
| Mean absolute error (MAE) | 2.901 points |
| Root mean squared error (RMSE) | 3.838 points |
| Mean signed error | +0.764 points |
| P10–P90 interval coverage | 75% |

The report describes this as a **top-three-per-fixture snapshot**, not full-squad accuracy. It is a limited early baseline, not evidence that the model is reliably accurate: the sample is small, all evaluated players currently have low confidence, and a few high-scoring outcomes dominate the errors.

## How to interpret it

- **MAE 2.901:** the average absolute gap between predicted and actual points is about 2.9 points for this small evaluated set.
- **Mean error +0.764:** the model overpredicted by 0.764 points on average in this sample. This is not yet enough to conclude a persistent bias.
- **P10–P90 coverage 75%:** three quarters of evaluated actual scores fell inside the model's stated interval. The nominal interval is broad, and this sample is too small to judge calibration.
- **Position metrics:** keep as diagnostics only until several full-pool gameweeks have accumulated.

## Next measurement gate

1. Keep existing GW8 results labelled as legacy/top-three-only; do not fabricate missing predictions after the deadline.
2. Ensure the next newly frozen snapshot includes the full eligible-player pool and an explicit `prediction_scope`, player count and snapshot timestamp.
3. After that gameweek finishes, compare its frozen predictions with official FPL points and verify name-match rate and eligible-player coverage before interpreting MAE/RMSE.
4. Build a rolling view across multiple completed gameweeks only once comparable full-pool records exist. Keep legacy and full-pool metrics separate.
5. Change model features only after the out-of-time comparison shows repeatable improvement.

## Guardrails

- Never replace missing predictions with zero.
- Never score a prediction generated after the deadline as if it were pre-deadline.
- Keep name-match rate and evaluated player count visible alongside accuracy.
- Do not call the P10–P90 interval a calibrated probability until multi-gameweek coverage has been tested.
