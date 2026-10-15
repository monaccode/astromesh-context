# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added

- Policies: `ScoredKnapsack` (density-ordered, per-source caps and floors, `min_items`),
  `FixedSlots` and `Everything`; pinned items always count against the budget.
- `dedupe`: exact (id, normalized hash) and near-duplicate (5-word shingles, Jaccard >= 0.8)
  removal before selection; pinned items never lose to non-pinned ones.
- `Calibrator`: per-source score calibration (blend, rank, minmax, zscore) with stale,
  human-reviewed and deprecated signal adjustments recorded for the trace.
- Project skeleton, data model (`ContextItem`, `Signals`, `Budget`, `Trace`, `Window`) and token
  counting (`CharEstimator`, `TiktokenTokenizer`).
