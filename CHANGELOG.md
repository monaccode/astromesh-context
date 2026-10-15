# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added

- `ContextEval` level 2 (`--level 2`): answers each window with a chat model and grades it against
  hand-written references (`answers.jsonl`) with an LLM judge; adds `accuracy`, `judge_unparsed`,
  p50 `latency_ms` and token columns, plus `openai_chat` and `judge_answer`. References are drafts
  pending review.
- Eval metrics `superseded_rate` and `noise_share`, a `candidate_tokens` column, and
  `config_meta` in the JSON report; the CLI default budgets are now 750, 1000 and 2000.
- The Nimbus OKF bundle marks each deprecated policy `superseded_by` its 2026 version.
- `ContextEval` level 1 and the `astromesh-context` CLI (`eval`, `explain`): compares
  `everything`, `fixed_slots` and `scored_knapsack` on the vendored Nimbus corpus (recall,
  precision per token, distractor rate, tokens, prefix reuse) and writes a dated JSON and
  Markdown report.
- `Offloader` (with `MemoryStore` / `DirStore`): moves oversized tool results out of the
  conversation, leaving a preview and a handle, and exposes `ctx_read` / `ctx_grep` tools to
  fetch what the agent needs.
- `Compactor` and `MessageAdapter` (`OpenAIAdapter`, `AnthropicAdapter`): summarize old turns
  without ever separating a tool call from its result; `units` is shared with `HistorySource`,
  which no longer groups a call without an id with an id-less tool message.
- `load_engine` / `load_engine_dict`: build a `ContextEngine` from a `kind: ContextPolicy` YAML
  (budget, tokenizer, policy, scorer, sources), with dotted `SpecError` paths for bad input.
- `RAGSource` (astromesh-rag pipeline or spec file), `OKFSource` (BM25 over a bundle's concepts,
  with trust/stale/deprecated signals) and `OKFIndexSource` (pinned bundle index), behind the
  `rag` and `okf` extras.
- `LLMScorer` and `openai_compatible`: optional model-based rescoring of the top candidates; any
  failure keeps the calibrated scores and records `scorer_fallback:<reason>`.
- `ContextEngine.build` with `StaticSource`, `CallableSource`, `HistorySource` and `SourceConfig`:
  concurrent source gathering with timeouts, graceful `source_error:` events, and a per-candidate
  `Trace`.
- Cache-stable ordering (`order_items`: stable, session, volatile edges-first, then history) and
  rendering to text, OpenAI and Anthropic messages, with `stable_prefix_hash`.
- Policies: `ScoredKnapsack` (density-ordered, per-source caps and floors, `min_items`),
  `FixedSlots` and `Everything`; pinned items always count against the budget.
- `dedupe`: exact (id, normalized hash) and near-duplicate (5-word shingles, Jaccard >= 0.8)
  removal before selection; pinned items never lose to non-pinned ones.
- `Calibrator`: per-source score calibration (blend, rank, minmax, zscore) with stale,
  human-reviewed and deprecated signal adjustments recorded for the trace.
- Project skeleton, data model (`ContextItem`, `Signals`, `Budget`, `Trace`, `Window`) and token
  counting (`CharEstimator`, `TiktokenTokenizer`).

### Changed

- Near-duplicate detection uses the containment coefficient over 5-word shingles instead of
  Jaccard, so a retrieval chunk matches the document it was cut from; texts with no words
  never near-match.
- `ScoredKnapsack` defaults to `beta=0.0` (calibrated-score order); `beta=0.5` density
  weighting favored short noise over long relevant documents on the Nimbus eval.

### Fixed

- Dedupe merges the dropped copy's signals (`deprecated`, `stale`, `trust`, `citation`) into
  the kept item, so a deprecation seen only on one copy is no longer lost.
