# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added

- `Calibrator`: per-source score calibration (blend, rank, minmax, zscore) with stale,
  human-reviewed and deprecated signal adjustments recorded for the trace.
- Project skeleton, data model (`ContextItem`, `Signals`, `Budget`, `Trace`, `Window`) and token
  counting (`CharEstimator`, `TiktokenTokenizer`).
