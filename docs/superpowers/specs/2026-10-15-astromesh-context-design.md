# astromesh-context v0.1 — design

**Date:** 2026-10-15 · **Status:** approved in conversation, pending written review
**Repo:** `monaccode/astromesh-context` (public, Apache-2.0) · **PyPI:** `astromesh-context` (name free)

## Goal

Decide what goes into an agent's context window, in one place, and measure the decision.

Retrieval ([astromesh-rag](https://github.com/monaccode/astromesh-rag)) and navigation
([astromesh-okf](https://github.com/monaccode/astromesh-okf)) decide what *could* be relevant.
Memory and the conversation add more. Nothing arbitrates between them: each stack spreads the
choice across prompt templates, a `top_k` and a history trim. This library is that arbiter, and
it ships with an eval so the claim "this window is better" comes with numbers.

It is the fifth family in the map the Astromesh docs publish
([The map of context patterns](https://monaccode.github.io/astromesh/context/patterns/)):
*what fits in the window?*

**Users**

1. **The community.** `pip install astromesh-context`, plug in sources, get a budgeted window with
   a trace, in any framework.
2. **The Astromesh core**, later. It replaces `MemoryManager.fit_history` and template prefetch.
   That migration is a separate spec, not part of v0.1.

**Success criteria**

- A user builds a window from RAG, OKF, history and static content in under 20 lines, and
  `window.trace.explain()` says why each candidate went in or stayed out.
- The window never exceeds the budget, pinned items are always present, and the same inputs
  always give the same window (property-tested).
- `ContextEval` level 1 runs in CI on the Nimbus corpus; level 2 produces the first published
  comparison table, with the command that regenerates it.
- The core depends only on PyYAML; RAG, OKF, tokenizers, LLM clients and frameworks are extras.

**Non-goals for v0.1**

- Graph sources, sub-agent isolation, framework integrations beyond the three listed.
- Migrating the Astromesh core (separate spec).
- Learning calibration weights from data. Weights are configured by hand in v0.1.
- Exact (DP) knapsack. Greedy by density is the algorithm; see §2.3.

## Decisions taken in conversation

| Question | Decision |
|---|---|
| What the library is | A context engine (family 5), not memory or graphs |
| When it acts | **Stateless core per call**; loop pieces (`Compactor`, `Offloader`) are optional and independent |
| How credibility is earned | An eval harness inside the library, with a published table |
| Selection approach | **Scored knapsack** as the core; **fixed slots** as a bundled baseline; **LLM selection** as an optional scoring step, not a policy |
| Name, visibility | `astromesh-context`, public from day one, marked "in design" until v0.1 |
| Python | `>=3.10`, matching astromesh-rag and astromesh-okf |

## Roadmap (each phase its own spec → plan → release)

- **v0.1 (this spec):** engine, three policies, calibration, dedupe, cache ordering, `LLMScorer`,
  sources (RAG, OKF index, OKF, history, static, callable), `Compactor`, `Offloader`,
  `ContextEval` (both levels), integrations for LangChain/LangGraph, PydanticAI and MCP,
  `kind: ContextPolicy` YAML.
- **v0.2:** remaining framework integrations (the set astromesh-okf supports); calibration
  weights fitted from an eval run.
- **Core migration (astromesh, next minor after v0.1):** `MemoryManager → Source` adapter,
  replacing `fit_history` and template prefetch.
- **Later:** graph source; sub-agent isolation helper.

## 1. Package and architecture

```
astromesh_context/
  __init__.py      public API re-exports
  types.py         ContextItem, Signals, Window, Trace, TraceEntry, Budget, Selection
  errors.py        ContextError, BudgetExceededError, SpecError
  source.py        Source protocol; StaticSource, CallableSource, HistorySource
  tokens.py        Tokenizer protocol; CharEstimator (default); TiktokenTokenizer ([tiktoken])
  calibrate.py     per-source normalization (rank, minmax, zscore) + signal adjustments
  scorer.py        Scorer protocol; LLMScorer ([llm] or any async callable)
  dedupe.py        exact (hash) and near-duplicate (shingle containment)
  policy.py        Policy protocol; ScoredKnapsack, FixedSlots, Everything
  order.py         cache-stable ordering
  engine.py        ContextEngine
  render.py        Window → OpenAI / Anthropic messages or a string
  spec.py          kind: ContextPolicy YAML → ContextEngine
  cli.py           astromesh-context eval | explain
  sources/
    rag.py         RAGSource                     ([rag])
    okf.py         OKFIndexSource, OKFSource     ([okf])
  loop/
    compactor.py   Compactor
    offloader.py   Offloader, MemoryStore, DirStore, ctx_read / ctx_grep
  eval/
    dataset.py     EvalDataset, nimbus() loader
    metrics.py     recall@budget, precision/token, distractor rate, prefix reuse
    judge.py       answer accuracy with an LLM judge (level 2)
    runner.py      ContextEval, Report
  integrations/
    langchain.py   ([langchain])
    pydantic_ai.py ([pydantic-ai])
    mcp.py         ([mcp])
```

**Boundaries**

- Sources are the only components that do I/O to fetch candidates. `LLMScorer` is the only
  other I/O, and it is optional.
- Calibration, dedupe, policy and ordering are pure functions of their inputs.
- `loop/` uses `types.py` and `tokens.py` only; it does not import `engine.py`.
- `eval/` uses the public API only, so it exercises what users get.

**Pipeline**

```
sources (async, parallel, per-source timeout)
  → calibrate (pure)
  → scorer (optional, async, falls back)
  → dedupe (pure)
  → policy.select (pure)
  → order (pure)
  → Window
```

## 2. Data model, selection and ordering

### 2.1 Contracts

```python
Kind = Literal["instruction", "knowledge", "memory", "history", "tool_result"]
Stability = Literal["stable", "session", "volatile"]

@dataclass(frozen=True)
class Signals:
    trust: str | None = None          # e.g. "human-reviewed", "unverified" (OKF vocabulary)
    stale: bool = False
    deprecated: bool = False
    citation: str | None = None       # concept id, doc id, URL

@dataclass(frozen=True)
class ContextItem:
    id: str                           # stable across calls for the same content
    content: str
    source: str                       # name of the Source that proposed it
    kind: Kind
    stability: Stability
    score: float = 0.0                # raw, source-specific
    tokens: int | None = None         # filled by the engine if None
    signals: Signals = Signals()
    pinned: bool = False
    order_key: str | None = None      # history: timestamp/sequence; else None
    meta: Mapping[str, Any] = field(default_factory=dict)  # source details for the trace (rag hit.details)

class Source(Protocol):
    name: str
    async def candidates(self, query: str, ctx: BuildContext) -> list[ContextItem]: ...

class Scorer(Protocol):
    async def score(self, query: str, items: Sequence[ScoredItem]) -> list[ScoredItem]: ...

class Policy(Protocol):
    def select(self, items: Sequence[ScoredItem], budget: Budget) -> Selection: ...
```

`Scorer.score` receives the calibrated items and returns them re-scored (`final` updated, its own
adjustment, e.g. `llm=<s>`, appended to `adjustments`), so the trace records the blend and the engine
does not need to know it. A sync `candidates` is accepted and run in a thread. `BuildContext` carries the budget, a
deadline and free-form `vars` for sources that need them (session id, user id).

`Window` holds `items` (ordered), `tokens_used`, `budget`, `trace` and `stable_prefix_hash`,
with `to_messages(style="openai"|"anthropic")` and `to_text()`.

`Trace` holds one `TraceEntry` per candidate: `item_id`, `source`, `raw_score`, `calibrated`,
`final`, `tokens`, `included: bool`, `reason` (see §4.1 for the closed list), plus build-level
events (`source_error`, `scorer_fallback`). `trace.explain()` renders it as a table.

### 2.2 Calibration

Per source, each item gets `cal = 0.5 · rank_score + 0.5 · minmax(raw)`, where
`rank_score = 1 − rank / n` (rank 0 is best). Sources with no numeric score set
`score` to `−rank`, and the formula reduces to rank. Alternatives `minmax` and `zscore` are
selectable per source.

Then:

- `cal *= source.weight` (default 1.0).
- Signal multipliers, configurable, defaults: `stale` ×0.7, `trust == "human-reviewed"` ×1.1.
- `deprecated` → excluded with reason `deprecated`, unless the policy sets
  `include_deprecated=True`. This matches `kb_find` in astromesh-okf.

Every adjustment is recorded in the trace entry.

### 2.3 Optional model scoring

`LLMScorer(complete, top_n=30, alpha=0.5, timeout=10.0)`:

- `complete` is any `async (prompt: str) -> str`; with the `[llm]` extra, a helper builds one
  from an OpenAI-compatible client.
- One batched call over the `top_n` items by calibrated score. The prompt lists numbered
  snippets and asks for a JSON array of 0–10 relevance scores.
- `final = alpha · (llm / 10) + (1 − alpha) · cal` for scored items; others keep `cal`.
- Failure, timeout, or a response that does not parse into exactly `top_n` numbers →
  `final = cal` for all, and a `scorer_fallback:<reason>` event.

### 2.4 Deduplication

1. Exact: SHA-256 of whitespace-normalized, lower-cased content.
2. Near: containment (overlap coefficient `|A∩B| / min(|A|,|B|)`) over 5-word shingles ≥ 0.8
   (configurable). Containment, not Jaccard: a retrieval chunk is a fragment of the document it
   came from, and symmetric Jaccard never reaches the threshold for fragment against whole.
   Texts with no words never near-match. O(n²) over candidates, which is
   fine for the hundreds this sees; documented as such.

The higher `final` wins; the other is excluded with `duplicate_of:<id>`. Pinned items always win.
History items are never deduplicated: a repeated turn is conversation, not redundant knowledge.

**Signal merge.** A dropped copy does not take its signals with it: they are merged into the
kept item. `deprecated` and `stale` are OR-ed; `trust` and `citation` are taken from the dropped
item only when the kept one has none. Each inherited signal adds an adjustment
`inherited:<signal>:<dropped id>` to the kept item's trace entry. So when a retrieval hit
outranks the OKF copy of a deprecated policy, the kept hit is still excluded as `deprecated`.

### 2.5 Policies

All three apply calibration, dedupe and ordering identically; they differ only in `select`.

**`ScoredKnapsack(beta=0.0, min_score=None, include_deprecated=False)`**

1. Pinned items in. If their tokens alone exceed the budget → `BudgetExceededError` listing
   each pinned item and its tokens.
2. Per-source `min_items` (e.g. history: the last user/assistant pair), best first, if they fit.
3. Remaining items by density `final / tokens^beta`, descending; add if it fits and the
   source is under its `max_tokens` cap and `final ≥ min_score`. Otherwise excluded with
   `budget`, `source_cap` or `below_floor`.
4. Back-fill: one pass over excluded-for-`budget` items in density order, adding any that now
   fit.

The default `beta=0` is pure calibrated-score order: on the Nimbus eval, density weighting
(`beta=0.5`) favored short noise over long relevant documents. `beta` stays a parameter and the
published table shows both.

Greedy, O(n log n), deterministic (ties broken by item id). Not exact: with noisy scores the
gap to an optimal knapsack is not worth the complexity, and the eval can measure it later.

**`FixedSlots(shares: dict[str, float])`**: each source gets `share · (budget − pinned)`
tokens, filled best-first; unused share is **not** redistributed (that is what makes it the
honest baseline). Shares must sum to ≤ 1.0.

**`Everything(truncate=False)`**: every non-deprecated candidate. Over budget → error, or with
`truncate=True` keep the best that fit and report what was cut.

### 2.6 Ordering for the cache

1. Group by stability: `stable` → `session` → `volatile`.
2. `stable`: sorted by `id`, so the bytes are identical whenever the set is.
3. `session`: by source order in the engine config, then `order_key`.
4. `volatile`: strongest at the edges of the block (1st, last, 2nd, second-to-last, …), the
   mitigation for lost-in-the-middle.
5. `history` items are rendered last and chronological regardless of stability.

`stable_prefix_hash` is the hash of the rendered stable block; the eval uses it for prefix reuse.

### 2.7 Tokens

`Tokenizer` protocol: `count(text) -> int`. Default `CharEstimator` (`ceil(len / 4)`), which is
documented as an estimate. `[tiktoken]` provides an exact counter for OpenAI encodings. The
budget is enforced against whichever tokenizer is configured; the docs say the estimate can
miss by ~10–20 % and recommend leaving headroom or using an exact counter.

### 2.8 YAML

```yaml
apiVersion: astromesh/v1
kind: ContextPolicy
metadata: { name: support }
spec:
  budget: 8000
  tokenizer: chars            # chars | tiktoken:<encoding>
  policy: { type: scored_knapsack, beta: 0.5 }
  scorer: { type: llm, top_n: 30, alpha: 0.5 }      # optional
  sources:
    - { type: static, text_file: prompts/system.md, kind: instruction, pinned: true, stability: stable }
    - { type: okf_index, bundle: ./kb, pinned: true }
    - { type: okf, bundle: ./kb, limit: 10, weight: 1.2 }
    - { type: rag, pipeline: pipelines/support.rag.yaml, k: 20, max_tokens: 3000 }
    - { type: history, min_items: 2 }
```

`policy` and `scorer` also accept a bare string (`policy: scored_knapsack`) meaning the type with
defaults. `load_engine(path, **runtime_objects)` builds the engine; objects that cannot live in YAML
(history messages, an LLM callable) are passed at load or build time. Unknown keys and invalid
values raise `SpecError` with the field path.

## 3. Sources and loop pieces

### 3.1 Sources

| Source | Extra | Emits | Raw score | kind / stability |
|---|---|---|---|---|
| `RAGSource(pipeline, k=20)` | `[rag]` | one item per `Hit` from `QueryPipeline.query()` | `hit.score`; `hit.details` copied to `meta` | knowledge / volatile |
| `OKFIndexSource(bundle, path="/")` | `[okf]` | `Bundle.index(path)` as one **pinned** item | — | instruction / stable |
| `OKFSource(bundle, limit=10)` | `[okf]` | concepts walked from the bundle with okf's rules and parsed with the public `parse_concept`, ranked by the source's own BM25 over title/description/tags (weight 2) and body (weight 1) | BM25 score | knowledge / volatile, `Signals` mapped from `trust`, `stale`, `status` |
| `HistorySource(messages)` | — | one item per turn; tool call + result kept as one item | recency | history / session; default `min_items=2` |
| `StaticSource(text, …)` | — | one item | given | configurable |
| `CallableSource(fn, …)` | — | whatever `fn(query, ctx)` returns | given | configurable |

`RAGSource` accepts an astromesh-rag `Pipeline`/`QueryPipeline` or a YAML path. Item ids are the
chunk id (`rag:<chunk_id>`), concept id (`okf:<id>`), and a stable hash for static content, so ids
are stable across calls and dedupe and caching work.

Memory facts and episodes enter through `CallableSource`. The Astromesh `MemoryManager`
adapter belongs to the core migration, not here.

### 3.2 `Compactor(summarize, threshold_tokens, keep_last=6, tokenizer=None)`

- Input and output: a list of messages. The Compactor works through a `MessageAdapter` protocol
  (role, text, tool-call ids, rebuild-with-text); adapters ship for OpenAI and Anthropic dicts and
  PydanticAI `ModelMessage` objects, so no conversion to dicts is lossy.
- Below threshold: returned unchanged (same object).
- Above: messages before the cut become one `summary` message; the last `keep_last` stay
  verbatim. The cut moves earlier until it does not separate an assistant tool call from its
  tool result(s).
- Incremental: an existing summary message is passed to `summarize` with the newly-aged turns,
  not re-summarized from scratch.
- `summarize` is `async (messages) -> str`; a default prompt (keep decisions, facts, ids,
  open questions) is provided for use with any completion callable.
- If `summarize` fails, the input is returned unchanged and the error is logged; compaction
  never drops content without a summary.

### 3.3 `Offloader(store, max_tokens=1500, preview_tokens=200)`

- Works through the same `MessageAdapter` as the Compactor. Scans tool-result messages; any over `max_tokens` is written to `store` and replaced by:
  the first `preview_tokens`, the handle, the total size, and one line telling the model to
  use `ctx_read` / `ctx_grep`.
- `ctx_read(handle, offset=0, limit=2000)` returns a character slice; `ctx_grep(handle,
  pattern, context=2, max_matches=20)` returns matching lines with context. Both **never
  raise**: an unknown handle or a bad regex returns an error string for the model.
- Stores: `MemoryStore` (default, process lifetime) and `DirStore(path)` (files; handle is a
  content hash, so the same result is stored once).

### 3.4 Integrations (v0.1)

- **LangChain/LangGraph (1.x):** `context_node(engine)` for a graph node before the model; an
  `AgentMiddleware` (`ContextMiddleware`) wrapping the compactor and offloader, replacing the
  pre-1.0 `pre_model_hook`; `offload_tools(store)` as LangChain tools.
- **PydanticAI:** `history_processor(compactor=…, offloader=…)`; `ctx_read`/`ctx_grep` as tools.
- **MCP:** `astromesh-context mcp --config policy.yaml` serving `ctx_build(query)`,
  `ctx_read`, `ctx_grep`.

Each integration is a thin adapter over the public API, with no logic of its own.

## 4. Errors, evaluation and testing

### 4.1 Errors

Principle: **a build degrades, never fails because of one source, and never trims silently.**

| Situation | Behaviour | Trace |
|---|---|---|
| A source raises or exceeds its timeout | Built without it | `source_error:<name>` with message |
| Every source fails | Window of pinned items only | per source |
| `LLMScorer` fails, times out, or returns unparseable output | Calibrated scores used | `scorer_fallback:<reason>` |
| Pinned items alone exceed the budget | `BudgetExceededError` with per-item tokens | — |
| `Everything` without `truncate` does not fit | `BudgetExceededError` | — |
| Invalid YAML spec | `SpecError` at load, with field path | — |
| Item without `tokens` | Counted with the configured tokenizer | — |

Exclusion reasons are a closed set: `budget`, `source_cap`, `below_floor`, `deprecated`,
`duplicate_of:<id>`, `share_exhausted` (FixedSlots), `truncated` (Everything).

All public exceptions subclass `ContextError`. Configuration errors fail early and loud;
runtime errors degrade and are recorded.

### 4.2 `ContextEval`

**Dataset: Nimbus.** The labeled corpus from astromesh-rag (200 documents with labeled traps,
60 questions with `relevant` doc ids). v0.1 vendors a pinned copy under `eval/data/nimbus/`
(with its license and source commit) so the eval does not depend on rag's repo layout, and
builds three sources from it:

- an **OKF bundle** generated from the 30 policy documents, with every `pol-*-2024` marked
  `status: deprecated`, `superseded_by: <2026 id>` and a closing link to its 2026 version;
- a **hybrid `RAGSource`** over all 200 documents (`HashEmbedder` for level 1; a real
  embedder for level 2);
- a **synthetic history** of N prior turns per question, for prefix reuse and history pressure.

**Level 1 — no model, deterministic, runs in CI**

- `recall@budget`: fraction of `relevant` docs present in the window.
- `precision_per_token`: tokens from relevant docs / window tokens.
- `distractor_rate`: windows containing a superseded policy or a `noise` doc. It saturates at
  generous budgets, so two finer metrics sit beside it: `superseded_rate` (windows with any
  `pol-*-2024` doc) and `noise_share` (tokens from `noise` docs / window tokens, averaged).
- `tokens_per_call`, and `candidate_tokens`: the mean size of the whole candidate pool (sum of
  tokens over every trace entry). Budgets must sit below the pool, or every policy selects
  the same items; on Nimbus the pool is about 3,200 tokens.
- `prefix_reuse`: over a simulated multi-turn session, the share of consecutive calls with an
  identical `stable_prefix_hash`, and the stable-block length.

**Level 2 — with a model, opt-in, for publishing**

- Answer accuracy judged by an LLM against a rubric and a reference answer.
- Latency and cost per call.
- Requires reference answers for the 60 questions, written by hand. This labeling is in scope
  for v0.1 and is committed alongside the corpus copy.

**The first published table:** `Everything(truncate=True)`, `FixedSlots`, `ScoredKnapsack`,
`ScoredKnapsack + LLMScorer`, at budgets 750, 1k and 2k (the pinned system prompt and OKF index alone take 512 tokens, so 500
cannot be run), both levels, with the exact command
and model ids. The table reports what comes out, including where `ScoredKnapsack` does not win.

```python
report = await ContextEval(dataset=nimbus(), configs={...}, budgets=[750, 1000, 2000]).run()
report.to_markdown(); report.to_json()
```

CLI: `astromesh-context eval --config eval.yaml [--level 2]`.

### 4.3 Testing

- **Pure unit tests** (the bulk): calibration, every signal adjustment, dedupe, each policy,
  ordering, every exclusion reason, rendering.
- **Properties** (`hypothesis`, dev dependency): window ≤ budget; pinned always present;
  deterministic for equal input; stable prefix identical when stable items are unchanged;
  `Compactor` never leaves a tool call without its result; `Offloader` round-trips content.
- **Source contracts** against a real in-memory astromesh-rag pipeline (`HashEmbedder`) and a
  fixture OKF bundle; skipped when the extras are absent, run in a dedicated CI job.
- **Injected failures:** raising source, slow source, `FakeLLM` returning garbage or too few
  scores.
- **Eval smoke:** level 1 over a Nimbus subset in CI, with floors that catch regressions
  (e.g. `ScoredKnapsack` recall@4k not below `FixedSlots`).
- **Integrations:** one test per framework with that framework's fake model.

**Tooling:** uv with a committed `uv.lock`, `ruff check` and `ruff format --check` (line length
100), pytest with `asyncio_mode = "auto"`, CI on Python 3.10–3.13, release through TestPyPI and
a smoke install before PyPI, as in astromesh-okf.

## 5. Documentation

- README (en/es) is already public and describes this design; it is updated to the shipped API
  when v0.1 lands, and the "in design" notice is removed then.
- Docs site: `context/introduction` and `context/patterns` exist; v0.1 adds `context/quickstart`,
  `context/policies`, `context/eval` (with the published table) and `context/integrations`, and
  the ecosystem entry moves from `inDevelopment` to a real version.
- `docs/` in the repo: `extending.md` (writing a Source, a Policy, a Scorer).
