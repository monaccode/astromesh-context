# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added

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
