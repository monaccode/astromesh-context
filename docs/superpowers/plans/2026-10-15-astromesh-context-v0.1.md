# astromesh-context v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `astromesh-context` 0.1.0: a stateless engine that builds an agent's context window from several sources under one token budget, with a trace, loop helpers, an eval on the Nimbus corpus, and three framework integrations.

**Architecture:** Sources (the only I/O) propose `ContextItem`s in parallel; pure stages calibrate, deduplicate, select (policy) and order them; an optional `LLMScorer` rescoring step falls back to calibrated scores. `loop/` (Compactor, Offloader) is independent of the engine and works through a `MessageAdapter`. `eval/` uses only the public API.

**Tech Stack:** Python ≥3.10, PyYAML, hatchling, uv, pytest + pytest-asyncio + hypothesis, ruff. Extras: astromesh-rag, astromesh-okf, tiktoken, httpx, langchain-core, pydantic-ai-slim, mcp.

**Spec:** `docs/superpowers/specs/2026-10-15-astromesh-context-design.md`

## Global Constraints

- Repo `D:\monaccode\astromesh-context`, branch `main`; package `astromesh_context`, distribution `astromesh-context`, version `0.1.0` in both `pyproject.toml` and `astromesh_context/__init__.py`.
- `requires-python = ">=3.10"`; core `dependencies = ["pyyaml>=6.0"]` and nothing else.
- Extras: `rag = ["astromesh-rag>=0.1,<0.2"]`, `okf = ["astromesh-okf>=0.1,<0.2"]`, `tiktoken = ["tiktoken>=0.7"]`, `llm = ["httpx>=0.27"]`, `langchain = ["langchain>=1.0", "langchain-core>=1.6"]`, `pydantic-ai = ["pydantic-ai-slim>=2.55; python_version >= '3.11'"]`, `mcp = ["mcp>=1.28"]`, `eval = ["astromesh-rag[sentence-transformers]>=0.1,<0.2", "httpx>=0.27"]`. astromesh-rag is not on PyPI yet: add `[tool.uv.sources] astromesh-rag = { git = "https://github.com/monaccode/astromesh-rag", tag = "v0.1.0" }` for development.
- Dev group: `pytest>=8`, `pytest-asyncio>=0.24`, `hypothesis>=6.100`, `ruff>=0.16`, `respx>=0.21`.
- `[tool.pytest.ini_options] testpaths = ["tests"]`, `asyncio_mode = "auto"`. `[tool.ruff]` `target-version = "py310"`, `line-length = 100`, lint `select` copied verbatim from `../astromesh-rag/pyproject.toml`.
- `uv.lock` committed; CI runs `uv sync --locked`. Before every commit: `uv run ruff check .` and `uv run ruff format --check .` (both must pass).
- The default test suite makes no network calls and loads no models.
- Conventional commits ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; every `feat:`/`fix:` commit adds a line under `## [Unreleased]` in `CHANGELOG.md` (Keep a Changelog).
- Exclusion reasons are exactly: `budget`, `source_cap`, `below_floor`, `deprecated`, `duplicate_of:<id>`, `share_exhausted`, `truncated`. Build events are exactly: `source_error:<name>:<message>`, `scorer_fallback:<reason>` with reason in `error|timeout|parse`.
- Defaults pinned by the spec: knapsack `beta=0.5`; stale ×0.7; `trust == "human-reviewed"` ×1.1; dedupe 5-word shingles, Jaccard ≥0.8; `LLMScorer(top_n=30, alpha=0.5, timeout=10.0)`; `Compactor(keep_last=6)`; `Offloader(max_tokens=1500, preview_tokens=200)`; `ctx_read(limit=2000)`; `ctx_grep(context=2, max_matches=20)`; `CharEstimator` = `ceil(len(text) / 4)`; per-source timeout `5.0` s; history `min_items=2`.

## Deviations from the spec (decided while planning; spec updated in Task 1)

1. **`OKFSource` scores concepts itself.** `Bundle.find(text=…)` is a literal substring match (a full question almost never matches) and caps at 100 results. `OKFSource` walks the bundle with okf's rules, parses with the public `parse_concept`, and ranks with BM25 over title/description/tags (weight 2) and body (weight 1). Raw score = BM25 score.
2. **LangChain 1.x uses middleware**, not `pre_model_hook`. The integration is an `AgentMiddleware`.
3. **Loop pieces work through a `MessageAdapter`**, so PydanticAI's `ModelMessage` objects are handled without lossy conversion to dicts.
4. **`Scorer.score` returns `list[ScoredItem]`** (spec §2.1 said `list[float]`), so the scorer records its own adjustment (`llm=<s>`) in the trace and the engine does not need to know the blend.

## Review Focus

1. **Empty or whitespace-only query** → sources still run; the window holds pinned items and history; no exception. Test in Task 6.
2. **An item with empty content (0 tokens)** → no division by zero in density; it counts as 1 token. Test in Task 4.
3. **Accented, non-English text** (the Nimbus corpus is mostly Spanish) → tokenization uses Unicode `\w+` lower-cased, so "devolución"/"DEVOLUCIÓN" dedupe and match in BM25. Tests in Tasks 3 and 8.
4. **Interrupted tool use in history** (assistant tool call with no result, or a result with no call) → `HistorySource` and `Compactor` keep the orphan as its own unit and never raise. Tests in Tasks 6 and 10.
5. **Two sources emitting the same item id** → the second is excluded `duplicate_of:<id>`, never silently overwritten. Test in Task 3.

---

## File Structure

```
astromesh_context/
  __init__.py        re-exports (Task 6 finalizes)
  errors.py          ContextError, BudgetExceededError, SpecError, MissingExtraError      T1
  types.py           Signals, ContextItem, ScoredItem, TraceEntry, Trace, Selection,
                     Budget, BuildContext, Window, REASONS                                  T1
  tokens.py          Tokenizer, CharEstimator, TiktokenTokenizer                           T1
  _text.py           words(), shingles(), content_hash()                                    T3
  calibrate.py       Calibrator                                                             T2
  dedupe.py          dedupe()                                                               T3
  policy.py          Policy, ScoredKnapsack, FixedSlots, Everything                         T4
  order.py           order_items()                                                          T5
  render.py          render_messages(), render_text(), stable_block()                       T5
  source.py          Source, SourceConfig, StaticSource, CallableSource, HistorySource      T6
  engine.py          ContextEngine                                                          T6
  scorer.py          Scorer, LLMScorer, openai_compatible()                                 T7
  sources/rag.py     RAGSource                                                              T8
  sources/okf.py     OKFIndexSource, OKFSource                                              T8
  spec.py            load_engine()                                                          T9
  loop/messages.py   MessageAdapter, OpenAIAdapter, AnthropicAdapter                        T10
  loop/compactor.py  Compactor                                                              T10
  loop/offloader.py  Offloader, MemoryStore, DirStore, ctx_read, ctx_grep                   T11
  eval/dataset.py    EvalCase, EvalDataset, nimbus(), nimbus_bundle()                       T12
  eval/metrics.py    recall_at_budget, precision_per_token, distractor_rate, prefix_reuse   T12
  eval/runner.py     ContextEval, Report                                                    T12
  eval/judge.py      judge_answer()                                                         T13
  eval/data/nimbus/  corpus.jsonl, queries.jsonl, answers.jsonl, SOURCE.md                  T12/T13
  cli.py             astromesh-context eval | explain | mcp                                 T12/T14
  integrations/      _base.py, langchain.py, pydantic_ai.py, mcp.py                         T14
tests/               one test module per source module; tests/integrations/ per framework
```

---

### Task 1: Project skeleton, data model, tokens

**Files:**
- Create: `pyproject.toml`, `CHANGELOG.md`, `CONTRIBUTING.md`, `astromesh_context/__init__.py`, `astromesh_context/py.typed`, `astromesh_context/errors.py`, `astromesh_context/types.py`, `astromesh_context/tokens.py`, `.github/workflows/ci.yml`, `uv.lock`
- Modify: `docs/superpowers/specs/2026-10-15-astromesh-context-design.md` (apply the four deviations above: §2.1 `Scorer`, §3.1 `OKFSource`, §3.4 LangChain, and the `MessageAdapter` in §3.2/§3.3)
- Test: `tests/test_types.py`, `tests/test_tokens.py`

**Interfaces:**
- Produces:
  - `class ContextError(Exception)`; `class BudgetExceededError(ContextError)` with attrs `budget: int`, `pinned: list[tuple[str, int]]`; `class SpecError(ContextError)` with attr `path: str`; `class MissingExtraError(ContextError)`.
  - `Kind = Literal["instruction","knowledge","memory","history","tool_result"]`, `Stability = Literal["stable","session","volatile"]`.
  - `@dataclass(frozen=True) Signals(trust: str|None=None, stale: bool=False, deprecated: bool=False, citation: str|None=None)`.
  - `@dataclass(frozen=True) ContextItem(id, content, source, kind, stability, score=0.0, tokens: int|None=None, signals=Signals(), pinned=False, order_key: str|None=None, meta: Mapping=field(default_factory=dict))` with `with_tokens(n) -> ContextItem` (via `dataclasses.replace`).
  - `@dataclass(frozen=True) ScoredItem(item: ContextItem, calibrated: float, final: float, adjustments: tuple[str, ...] = ())`; property `tokens -> int` returning `max(1, item.tokens or 0)`.
  - `@dataclass(frozen=True) Budget(total: int, caps: Mapping[str,int] = {}, min_items: Mapping[str,int] = {})` (use `field(default_factory=dict)`); `__post_init__` raises `ValueError` if `total <= 0`.
  - `@dataclass Selection(included: list[ScoredItem], excluded: list[tuple[ScoredItem, str]])`.
  - `@dataclass(frozen=True) TraceEntry(item_id, source, raw_score: float, calibrated: float, final: float, tokens: int, included: bool, reason: str|None, adjustments: tuple[str,...])`.
  - `@dataclass Trace(entries: list[TraceEntry], events: list[str])` with `explain() -> str` (fixed-width table: id, source, final (3 decimals), tokens, `in`/`out`, reason; events listed after) and `excluded(reason_prefix: str) -> list[TraceEntry]`.
  - `@dataclass BuildContext(query: str, budget: Budget, deadline: float, vars: Mapping[str, Any])`.
  - `Window` is declared here as a dataclass `(query: str, items: list[ContextItem], tokens_used: int, budget: int, trace: Trace, stable_prefix_hash: str)`; its `to_messages`/`to_text` methods are added in Task 5.
  - `class Tokenizer(Protocol): def count(self, text: str) -> int`; `CharEstimator().count` = `math.ceil(len(text)/4)`; `TiktokenTokenizer(encoding="cl100k_base")` imports `tiktoken` lazily and raises `MissingExtraError("install astromesh-context[tiktoken]")` when absent.

- [ ] **Step 1: Write `pyproject.toml`** per Global Constraints (sdist include/exclude and wheel packages mirroring `../astromesh-rag/pyproject.toml`; `[project.scripts] astromesh-context = "astromesh_context.cli:main"`). `CHANGELOG.md` with `## [Unreleased]`. Run `uv lock`. Expected: `uv.lock` created.
- [ ] **Step 2: Write failing tests**

```python
def test_budget_rejects_non_positive():
    with pytest.raises(ValueError): Budget(0)

def test_scored_item_tokens_floor_is_one():
    it = ContextItem("a", "", "s", "knowledge", "volatile", tokens=0)
    assert ScoredItem(it, 0.5, 0.5).tokens == 1

def test_context_item_is_frozen_and_with_tokens_copies():
    it = ContextItem("a", "hello", "s", "knowledge", "volatile")
    assert it.with_tokens(3).tokens == 3 and it.tokens is None

def test_trace_explain_lists_reason_and_events():
    t = Trace([TraceEntry("a","s",1.0,0.9,0.9,10,False,"budget",())], ["source_error:x:boom"])
    out = t.explain()
    assert "budget" in out and "source_error:x:boom" in out

def test_char_estimator():
    assert CharEstimator().count("abcde") == 2 and CharEstimator().count("") == 0

def test_tiktoken_missing_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    with pytest.raises(MissingExtraError): TiktokenTokenizer().count("x")
```

- [ ] **Step 3: Run** `uv run pytest tests/test_types.py tests/test_tokens.py -q`. Expected: FAIL (import errors).
- [ ] **Step 4: Implement** `errors.py`, `types.py`, `tokens.py` with the signatures above. `__init__.py` defines `__version__ = "0.1.0"`.
- [ ] **Step 5: Run** the tests again plus `uv run ruff check . && uv run ruff format --check .`. Expected: PASS, clean.
- [ ] **Step 6: Write `.github/workflows/ci.yml`**: jobs `lint` and `core` (matrix 3.10–3.13) copied from `../astromesh-okf/.github/workflows/ci.yml`; an empty `extras` job is added in Task 8. Apply the spec deviations edit.
- [ ] **Step 7: Commit** `feat: project skeleton, data model and token counting` (with CHANGELOG line).

---

### Task 2: Calibration

**Files:** Create `astromesh_context/calibrate.py`; Test `tests/test_calibrate.py`

**Interfaces:**
- Consumes: `ContextItem`, `ScoredItem` (T1).
- Produces: `class Calibrator(stale: float = 0.7, human_reviewed: float = 1.1)` with `calibrate(items: Sequence[ContextItem], *, weights: Mapping[str, float] = {}, modes: Mapping[str, str] = {}) -> list[ScoredItem]`. Modes per source: `"blend"` (default), `"rank"`, `"minmax"`, `"zscore"`.

Algorithm, per source group (items keep their input order in the output):

```
rank r: 0 = highest raw score; ties broken by item id
rank_score = 1 - r / n                      (n = group size; a single item gets 1.0)
minmax     = (raw - min) / (max - min), or 1.0 when max == min
zscore     = 1 / (1 + exp(-(raw - mean) / std)), or 0.5 when std == 0
blend      = 0.5 * rank_score + 0.5 * minmax
cal = mode_value * weights.get(source, 1.0)
stale → cal *= stale, adjustment "stale×0.7"
trust == "human-reviewed" → cal *= human_reviewed, adjustment "human-reviewed×1.1"
deprecated → adjustment "deprecated" (exclusion happens in the policy)
pinned → cal = 1.0, no adjustments
final = cal
```

- [ ] **Step 1: Write failing tests**: `test_blend_two_items` (raw 0.9 and 0.5 in one source → cals `1.0` and `0.25`); `test_equal_scores_minmax_is_one`; `test_sources_calibrated_independently` (BM25-scale 14.0 in one source and cosine 0.8 in another both get `1.0` as the top of their source); `test_weight_multiplies`; `test_stale_and_human_reviewed_adjust_and_record` (asserts `adjustments == ("stale×0.7",)` etc.); `test_deprecated_recorded_not_removed`; `test_pinned_is_one`; `test_rank_ties_broken_by_id`.
- [ ] **Step 2: Run** `uv run pytest tests/test_calibrate.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `Calibrator.calibrate`.
- [ ] **Step 4: Run** tests and ruff. Expected: PASS.
- [ ] **Step 5: Commit** `feat: per-source score calibration with signal adjustments`.

---

### Task 3: Deduplication

**Files:** Create `astromesh_context/_text.py`, `astromesh_context/dedupe.py`; Test `tests/test_dedupe.py`

**Interfaces:**
- Produces:
  - `_text.words(text) -> list[str]` = `re.findall(r"\w+", text.lower())` (Unicode); `_text.shingles(text, n=5) -> frozenset[tuple[str, ...]]` (texts with fewer than `n` words yield one shingle of all their words); `_text.content_hash(text) -> str` = sha256 of `" ".join(text.lower().split())`.
  - `dedupe(items: Sequence[ScoredItem], threshold: float = 0.8) -> tuple[list[ScoredItem], list[tuple[ScoredItem, str]]]`. Process by `final` descending (pinned first, ties by id). An item is dropped with `duplicate_of:<kept id>` when its id equals a kept id, its `content_hash` equals a kept one, or Jaccard of shingles ≥ `threshold`. Kept items return in input order.

- [ ] **Step 1: Write failing tests**: `test_exact_duplicate_keeps_higher_final`; `test_whitespace_and_case_insensitive_exact`; `test_near_duplicate_over_threshold` (two 30-word texts differing in one word); `test_below_threshold_both_kept`; `test_pinned_always_wins` (pinned with lower final keeps, other gets `duplicate_of:<pinned id>`); `test_same_id_from_two_sources` (Review Focus 5); `test_accented_text_dedupes` ("La DEVOLUCIÓN se procesa en 30 días hábiles…" vs lower-case copy) (Review Focus 3).
- [ ] **Step 2: Run** `uv run pytest tests/test_dedupe.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `_text.py` and `dedupe.py`.
- [ ] **Step 4: Run** tests and ruff. Expected: PASS.
- [ ] **Step 5: Commit** `feat: exact and near-duplicate removal`.

---

### Task 4: Policies

**Files:** Create `astromesh_context/policy.py`; Test `tests/test_policy.py`

**Interfaces:**
- Consumes: `ScoredItem`, `Budget`, `Selection`, `BudgetExceededError` (T1).
- Produces:
  - `class Policy(Protocol): def select(self, items: Sequence[ScoredItem], budget: Budget) -> Selection`.
  - `ScoredKnapsack(beta: float = 0.5, min_score: float | None = None, include_deprecated: bool = False)`.
  - `FixedSlots(shares: Mapping[str, float], include_deprecated: bool = False)`; `ValueError` if any share < 0 or the sum > 1.0.
  - `Everything(truncate: bool = False, include_deprecated: bool = False)`.
  - Module helper `_split_pinned(items, budget) -> tuple[list, list, int]` shared by all three: removes deprecated (unless allowed) with reason `deprecated`, returns pinned, rest and pinned tokens; raises `BudgetExceededError(budget.total, [(id, tokens), …])` when pinned tokens > `budget.total`.

`ScoredKnapsack.select`, after `_split_pinned`:
1. For each source in `budget.min_items`, take its best `min_items[source]` by `final` (ties by id) if each fits the remaining budget.
2. Remaining items sorted by `final / tokens**beta` descending, ties by id. Add if `final >= min_score` (else `below_floor`), source tokens + item ≤ `caps[source]` (else `source_cap`), and it fits (else `budget`).
3. Back-fill: one pass over `budget` exclusions in the same order, adding any that now fit.

`FixedSlots.select`: each source gets `floor(share * (total - pinned_tokens))` tokens; fill best-first by `final`; items that do not fit their share, or whose source has no share, get `share_exhausted`. Unused share is not redistributed.

`Everything.select`: all remaining items; if they exceed the budget, raise `BudgetExceededError` unless `truncate`, in which case keep best-first by `final` and mark the rest `truncated`.

- [ ] **Step 1: Write failing tests**: `test_pinned_over_budget_raises_with_items`; `test_knapsack_prefers_density` (A: final 0.9, 900 tokens; B and C: final 0.6, 100 tokens each; budget 250 → B and C in, A `budget`); `test_knapsack_min_items_forced` (history items with low final included first); `test_knapsack_source_cap`; `test_knapsack_below_floor`; `test_knapsack_backfill`; `test_deprecated_excluded_unless_allowed`; `test_zero_token_item_counts_as_one` (Review Focus 2); `test_fixed_slots_does_not_redistribute` (rag share 0.5 unused, history overflows → `share_exhausted`); `test_fixed_slots_rejects_shares_over_one`; `test_everything_raises_when_over`; `test_everything_truncate_marks_truncated`; `test_select_is_deterministic` (shuffled input → identical `Selection`).
- [ ] **Step 2: Run** `uv run pytest tests/test_policy.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `policy.py`.
- [ ] **Step 4: Run** tests and ruff. Expected: PASS.
- [ ] **Step 5: Commit** `feat: scored knapsack, fixed slots and everything policies`.

---

### Task 5: Ordering and rendering

**Files:** Create `astromesh_context/order.py`, `astromesh_context/render.py`; Modify `astromesh_context/types.py` (add `Window.to_messages`, `Window.to_text`); Test `tests/test_order.py`, `tests/test_render.py`

**Interfaces:**
- Consumes: `ScoredItem`, `ContextItem`, `Window` (T1).
- Produces:
  - `order_items(items: Sequence[ScoredItem], source_order: Sequence[str]) -> list[ContextItem]`: non-history first, grouped `stable → session → volatile`; stable sorted by id; session by `source_order` index then `order_key` then id; volatile by `final` descending placed edges-first (positions 0, −1, 1, −2, …). Then all `kind == "history"` items by `order_key`.
  - `stable_block(items: Sequence[ContextItem]) -> str` and `render_text(items) -> str`: each non-history item is rendered as `<context id="{id}" source="{source}">\n{content}\n</context>`, except `kind == "instruction"`, which is rendered raw; blocks joined by `"\n\n"`. `stable_block` renders only the `stable` items.
  - `render_messages(items, query, style: Literal["openai","anthropic"]) -> list[dict] | dict`: openai → `[{"role":"system","content": render_text(non-history)}, *history messages, {"role":"user","content": query}]`; anthropic → `{"system": render_text(...), "messages": [*history messages, {"role":"user","content": query}]}`. History messages come from `item.meta["messages"]` (set by `HistorySource`). An empty query adds no user message.
  - `Window.to_messages(style="openai", include_query=True)`, `Window.to_text()`; `stable_prefix_hash` = sha256 hex of `stable_block(...)`.

- [ ] **Step 1: Write failing tests**: `test_stability_groups_in_order`; `test_stable_sorted_by_id_regardless_of_input`; `test_volatile_edges_first` (finals 5,4,3,2,1 → order 5,3,1,2,4); `test_history_last_and_chronological`; `test_instruction_rendered_raw_knowledge_wrapped`; `test_openai_messages_shape`; `test_anthropic_messages_shape`; `test_stable_hash_unchanged_when_only_volatile_changes`.
- [ ] **Step 2: Run** `uv run pytest tests/test_order.py tests/test_render.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `order.py`, `render.py` and the `Window` methods.
- [ ] **Step 4: Run** tests and ruff. Expected: PASS.
- [ ] **Step 5: Commit** `feat: cache-stable ordering and message rendering`.

---

### Task 6: Sources and the engine

**Files:** Create `astromesh_context/source.py`, `astromesh_context/engine.py`; Modify `astromesh_context/__init__.py`; Test `tests/test_sources.py`, `tests/test_engine.py`, `tests/test_properties.py`

**Interfaces:**
- Consumes: everything from T1–T5.
- Produces:
  - `class Source(Protocol)`: `name: str`; `candidates(query: str, ctx: BuildContext) -> list[ContextItem] | Awaitable[list[ContextItem]]`.
  - `@dataclass SourceConfig(source: Source, weight: float = 1.0, max_tokens: int | None = None, min_items: int = 0, timeout: float | None = None, mode: str = "blend")`.
  - `StaticSource(text: str, *, name: str = "static", kind: Kind = "instruction", stability: Stability = "stable", pinned: bool = False, score: float = 1.0)`; item id `static:<content_hash[:12]>`.
  - `CallableSource(fn: Callable[[str, BuildContext], list[ContextItem] | Awaitable[list[ContextItem]]], *, name: str)`.
  - `HistorySource(messages: list[dict] | None = None, *, key: str = "history", name: str = "history")`: uses `messages`, else `ctx.vars.get(key, [])`. One item per unit: a user or assistant message, or an assistant message with `tool_calls` plus every following `role == "tool"` message whose `tool_call_id` it declared. Orphans form their own unit. Id `history:<index of first message>`; `order_key` = that index zero-padded to 8 digits; `score` = index; `kind="history"`, `stability="session"`; `meta["messages"]` = the unit's messages; content = the messages' text joined.
  - `ContextEngine(sources: Sequence[Source | SourceConfig], *, budget: int, policy: Policy | None = None, tokenizer: Tokenizer | None = None, calibrator: Calibrator | None = None, dedupe_threshold: float = 0.8, source_timeout: float = 5.0)`; `async build(query: str, *, vars: Mapping[str, Any] | None = None) -> Window`. Defaults: `ScoredKnapsack()`, `CharEstimator()`, `Calibrator()`. A `HistorySource` without an explicit `SourceConfig` gets `min_items=2`. `Budget.caps` and `Budget.min_items` are built from the `SourceConfig`s, keyed by source name.
  - `build` steps: gather `candidates` concurrently, each under `asyncio.wait_for(timeout)`, running sync results in `asyncio.to_thread`; a failure records `source_error:<name>:<message>` and drops that source → fill `tokens` with the tokenizer → `Calibrator.calibrate` → `dedupe` → `policy.select` → `order_items` → build `Trace` (one entry per candidate, in candidate order) → `Window`.
  - `__init__.py` re-exports: `ContextEngine, ContextItem, Signals, Window, Trace, Budget, SourceConfig, StaticSource, CallableSource, HistorySource, ScoredKnapsack, FixedSlots, Everything, Calibrator, CharEstimator, ContextError, BudgetExceededError, SpecError, MissingExtraError, __version__`.

- [ ] **Step 1: Write failing source tests**: `test_static_source_stable_id`; `test_callable_source_sync_and_async`; `test_history_pairs_tool_call_with_results`; `test_history_orphan_tool_call_is_own_unit` and `test_history_orphan_tool_result_is_own_unit` (Review Focus 4); `test_history_reads_ctx_vars`.
- [ ] **Step 2: Write failing engine tests**: `test_build_end_to_end` (static pinned + callable knowledge + history, budget 200 → pinned present, newest pair present, `tokens_used <= 200`, every candidate in the trace); `test_failing_source_recorded_and_skipped`; `test_slow_source_times_out` (source sleeps 1 s, `source_timeout=0.05` → `source_error:slow:` event); `test_all_sources_fail_returns_pinned_only`; `test_empty_query` (Review Focus 1); `test_pinned_over_budget_raises`.
- [ ] **Step 3: Write property tests** with `hypothesis` (`@given` random lists of items with tokens 0–500, scores, stabilities, pinned flags kept under budget): `window.tokens_used <= budget`; every pinned item is in `window.items`; two builds with the same input give equal `[i.id for i in window.items]`; `stable_prefix_hash` equal across builds that change only volatile items.
- [ ] **Step 4: Run** `uv run pytest tests/test_sources.py tests/test_engine.py tests/test_properties.py -q`. Expected: FAIL.
- [ ] **Step 5: Implement** `source.py`, `engine.py`, `__init__.py`.
- [ ] **Step 6: Run** the full suite and ruff. Expected: PASS.
- [ ] **Step 7: Commit** `feat: sources and ContextEngine with trace and graceful source failure`.

---

### Task 7: LLMScorer

**Files:** Create `astromesh_context/scorer.py`; Modify `astromesh_context/engine.py` (add the `scorer: Scorer | None = None` keyword to `ContextEngine.__init__` and the scoring step), `astromesh_context/__init__.py` (export `LLMScorer`); Test `tests/test_scorer.py`

**Interfaces:**
- Consumes: `ScoredItem`, `ContextEngine` (T6).
- Produces:
  - `class Scorer(Protocol): async def score(self, query: str, items: Sequence[ScoredItem]) -> list[ScoredItem]`.
  - `class ScorerFallback(Exception)` with attr `reason: Literal["error","timeout","parse"]`.
  - `LLMScorer(complete: Callable[[str], Awaitable[str]], *, top_n: int = 30, alpha: float = 0.5, timeout: float = 10.0, max_chars: int = 800)`. `score` takes the `top_n` non-pinned items by `final`, sends one prompt, and returns all items with `final = alpha * clamp(s, 0, 10) / 10 + (1 - alpha) * calibrated` for the scored ones (adjustment `llm=<s>`). It raises `ScorerFallback` on exception, timeout, or a response whose first JSON array does not hold exactly as many numbers as snippets.
  - Prompt (exact copy): `"Rate how useful each numbered snippet is for answering the question. Reply with only a JSON array of numbers from 0 to 10, one per snippet, in order.\n\nQuestion: {query}\n\n{snippets}"`, each snippet `"[{i}] {content[:max_chars]}"` joined by `"\n\n"`.
  - `openai_compatible(base_url: str, api_key: str, model: str, *, timeout: float = 30.0) -> Callable[[str], Awaitable[str]]`: POST `{base_url}/chat/completions` with httpx (`[llm]` extra, `MissingExtraError` otherwise), `temperature=0`, returns `choices[0].message.content`.
  - Engine: on `ScorerFallback(reason)`, keep the calibrated items and record `scorer_fallback:<reason>`.

- [ ] **Step 1: Write failing tests** with a fake `complete`: `test_blends_scores` (calibrated 0.4, llm 10, alpha 0.5 → final 0.7); `test_only_top_n_sent` (count snippets in the captured prompt); `test_pinned_not_sent`; `test_garbage_response_falls_back` (engine trace has `scorer_fallback:parse`); `test_wrong_count_falls_back`; `test_timeout_falls_back` (complete sleeps; `timeout=0.05`); `test_exception_falls_back`; `test_openai_compatible_posts` with `respx` (asserts URL, `model`, `temperature == 0`, and that the returned string is the message content).
- [ ] **Step 2: Run** `uv run pytest tests/test_scorer.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `scorer.py` and the engine step (between calibration and dedupe).
- [ ] **Step 4: Run** the full suite and ruff. Expected: PASS.
- [ ] **Step 5: Commit** `feat: optional LLM rescoring with calibrated fallback`.

---

### Task 8: RAG and OKF sources

**Files:** Create `astromesh_context/sources/__init__.py`, `astromesh_context/sources/rag.py`, `astromesh_context/sources/okf.py`, `astromesh_context/_bm25.py`, `tests/fixtures/kb/` (6 concepts: two policies where one is `status: deprecated`, one `stale_after` in the past, one `verified` by a human, one `index.md`, one `log.md`); Modify `.github/workflows/ci.yml` (extras job); Test `tests/test_source_rag.py`, `tests/test_source_okf.py`

**Interfaces:**
- Consumes: `ContextItem`, `Signals`, `BuildContext` (T1); astromesh-rag `Pipeline`/`QueryPipeline` (`async query(q, k) -> QueryResult` with `.hits[*].chunk.{id, doc_id, text}`, `.score`, `.details`) and `astromesh_rag.spec.build`, `load_spec_file`; astromesh-okf `Bundle(root)`, `Bundle.index(path)`, `parse_concept(root, path, now)`, `InvalidConceptError`, `astromesh_okf.concept.RESERVED`.
- Produces:
  - `_bm25.BM25(k1=1.5, b=0.75)` with `fit(docs: list[list[str]])`, `scores(query_terms: list[str]) -> list[float]`.
  - `RAGSource(pipeline, *, k: int = 20, name: str = "rag")`: `pipeline` may be a str/Path (loaded with `build(load_spec_file(path))`; needs the `[rag]` extra) or any object with `async query(q, k)`; the object form needs no import of astromesh_rag. Item: id `rag:<chunk.id>`, `content=chunk.text`, `score=hit.score`, `kind="knowledge"`, `stability="volatile"`, `signals=Signals(citation=chunk.doc_id)`, `meta={"doc_id": chunk.doc_id, **hit.details}`.
  - `OKFIndexSource(bundle: str | Path, *, path: str = "/", name: str = "okf_index")`: one item, id `okf_index:<path>`, `kind="instruction"`, `stability="stable"`, `pinned=True`, content `Bundle(root).index(path)`.
  - `OKFSource(bundle: str | Path, *, limit: int = 10, name: str = "okf")`: walk `root.rglob("*.md")` skipping `RESERVED`, dot-paths, and paths resolving outside root; parse each with `parse_concept`, skipping `InvalidConceptError`; BM25 with the header (title, description, tags) counted twice; keep the top `limit` with score > 0, deprecated included. Item: id `okf:<concept.id>`, content `"# {title}\n\n{body}"`, `score` = BM25 score, `kind="knowledge"`, `stability="volatile"`, `signals=Signals(trust=concept.trust, stale=concept.stale, deprecated=concept.status == "deprecated", citation=concept.id)`, `meta={"concept_id": concept.id}`. Parsing runs in `asyncio.to_thread`. A missing `astromesh_okf` raises `MissingExtraError("install astromesh-context[okf]")` at construction.

- [ ] **Step 1: Write failing RAG tests** with a real in-memory astromesh-rag pipeline (`IngestPipeline(RecursiveSplitter(800, 80), HashEmbedder(), MemoryIndex(), BM25Index())` + `hybrid(...)`, three documents): `test_rag_items_carry_doc_id_and_details`; `test_rag_duck_typed_pipeline_needs_no_extra` (a fake object with `async query`); module-level `pytest.importorskip("astromesh_rag")` on the real-pipeline tests.
- [ ] **Step 2: Write failing OKF tests** against `tests/fixtures/kb`: `test_index_source_pinned_stable`; `test_okf_ranks_by_terms` ("refund days" ranks the refund policy first); `test_okf_signals_mapped` (deprecated, stale, trust); `test_okf_skips_reserved_and_invalid`; `test_okf_accent_insensitive_case` ("DEVOLUCIÓN" matches "devolución") (Review Focus 3); `test_okf_missing_extra` (monkeypatch `sys.modules["astromesh_okf"] = None`).
- [ ] **Step 3: Run** `uv sync --extra rag --extra okf` then `uv run pytest tests/test_source_rag.py tests/test_source_okf.py -q`. Expected: FAIL.
- [ ] **Step 4: Implement** `_bm25.py`, `sources/rag.py`, `sources/okf.py`.
- [ ] **Step 5: Add a CI `extras` job** (as in okf) with matrix entries `{extra: rag, test: test_source_rag.py}` and `{extra: okf, test: test_source_okf.py}`; the `core` job runs `pytest -q --ignore` on those two files.
- [ ] **Step 6: Run** the full suite with extras and ruff; then run `uv lock` and check that `uv sync --locked` succeeds. Expected: PASS.
- [ ] **Step 7: Commit** `feat: RAG and OKF sources`.

---

### Task 9: YAML spec

**Files:** Create `astromesh_context/spec.py`, `tests/fixtures/policy.yaml`, `tests/fixtures/system.md`; Test `tests/test_spec.py`

**Interfaces:**
- Consumes: `ContextEngine`, all sources, policies, `LLMScorer`, tokenizers, `SpecError`.
- Produces: `load_engine(path: str | Path, *, complete: Callable[[str], Awaitable[str]] | None = None, objects: Mapping[str, Any] = {}) -> ContextEngine`, and `load_engine_dict(raw: dict, *, base_dir: Path, complete=None, objects={}) -> ContextEngine`.
  - Top level must be `apiVersion: astromesh/v1`, `kind: ContextPolicy`; `spec.budget` (int > 0) is required.
  - `tokenizer`: `chars` (default) or `tiktoken:<encoding>`.
  - `policy`: string or `{type, ...params}`; types `scored_knapsack`, `fixed_slots`, `everything`.
  - `scorer`: string or `{type: llm, top_n, alpha, timeout}`; requires `complete`, else `SpecError` at `spec.scorer`.
  - Each source: `type` ∈ `static` (`text` or `text_file`, resolved against `base_dir`), `okf_index` (`bundle`, `path`), `okf` (`bundle`, `limit`), `rag` (`pipeline`: a path, or `object: <key>` looked up in `objects`), `history` (`key`), `callable` (`object: <key>`). Shared keys: `name`, `weight`, `max_tokens`, `min_items`, `timeout`, `mode`, `pinned`, `kind`, `stability`.
  - Unknown keys or bad values → `SpecError` whose `path` is dotted, e.g. `spec.sources[2].limit`.

- [ ] **Step 1: Write failing tests**: `test_loads_full_fixture` (asserts source names, budget, policy type); `test_policy_string_shorthand`; `test_unknown_key_reports_path` (`spec.sources[1].lmit`); `test_wrong_kind_rejected`; `test_scorer_without_complete_rejected`; `test_text_file_relative_to_yaml`; `test_object_reference_missing_rejected`.
- [ ] **Step 2: Run** `uv run pytest tests/test_spec.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `spec.py` and export `load_engine` from `__init__`.
- [ ] **Step 4: Run** the full suite and ruff. Expected: PASS.
- [ ] **Step 5: Commit** `feat: kind ContextPolicy YAML loader`.

---

### Task 10: Message adapters and Compactor

**Files:** Create `astromesh_context/loop/__init__.py`, `astromesh_context/loop/messages.py`, `astromesh_context/loop/compactor.py`; Test `tests/test_messages.py`, `tests/test_compactor.py`

**Interfaces:**
- Consumes: `Tokenizer`, `CharEstimator` (T1).
- Produces:
  - `class MessageAdapter(Protocol[M])`: `text(m: M) -> str`; `tool_call_ids(m: M) -> list[str]` (ids an assistant message calls); `tool_result_id(m: M) -> str | None`; `is_summary(m: M) -> bool`; `make_summary(text: str) -> M`; `replace_tool_result(m: M, text: str) -> M`.
  - `OpenAIAdapter` (dict messages: `tool_calls[*].id`, `role == "tool"` + `tool_call_id`) and `AnthropicAdapter` (content blocks `tool_use.id` / `tool_result.tool_use_id`). A summary is a message with `"name": "context_summary"` (OpenAI, role `system`) or a user message whose first text block starts with `"[context summary]"` (Anthropic).
  - `units(messages, adapter) -> list[list[M]]`: the grouping rule from `HistorySource` (an assistant call plus its results; orphans alone). `HistorySource` is refactored to use it with `OpenAIAdapter`.
  - `Compactor(summarize: Callable[[list[M]], Awaitable[str]], *, threshold_tokens: int, keep_last: int = 6, adapter: MessageAdapter = OpenAIAdapter(), tokenizer: Tokenizer | None = None)`; `async compact(messages: list[M]) -> list[M]`.
  - `compact` algorithm: if the total tokens ≤ threshold, return the same list object. Otherwise the kept tail is the smallest suffix of whole units holding ≥ `keep_last` messages; leading system messages that are not summaries stay first; an existing summary message plus the aged units go to `summarize`; the result is `[*leading system, make_summary(text), *tail]`. If `summarize` raises, log a warning via `logging.getLogger("astromesh_context")` and return the input unchanged.
  - `DEFAULT_SUMMARY_PROMPT` (exact copy): `"Summarize the conversation so far for the assistant that will continue it. Keep every decision, fact, number, identifier and open question. Drop greetings and repetition. If a previous summary is included, merge it; do not summarize the summary."` plus `summary_prompt(messages, adapter) -> str` that appends the rendered messages.

- [ ] **Step 1: Write failing tests**: adapter round-trips for both styles; `test_under_threshold_returns_same_object`; `test_keeps_last_and_summarizes_rest`; `test_never_splits_tool_call_from_results` (cut point lands inside a pair → moves earlier); `test_incremental_includes_previous_summary_once`; `test_leading_system_preserved`; `test_summarize_failure_returns_input`; `test_orphan_tool_call_survives` (Review Focus 4); property test: for random OpenAI histories, every `tool_call_id` in the output is answered or was an orphan in the input.
- [ ] **Step 2: Run** `uv run pytest tests/test_messages.py tests/test_compactor.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `messages.py` and `compactor.py`; refactor `HistorySource` onto `units`.
- [ ] **Step 4: Run** the full suite and ruff. Expected: PASS.
- [ ] **Step 5: Commit** `feat: message adapters and tool-safe history compaction`.

---

### Task 11: Offloader

**Files:** Create `astromesh_context/loop/offloader.py`; Test `tests/test_offloader.py`

**Interfaces:**
- Consumes: `MessageAdapter`, `OpenAIAdapter` (T10), `Tokenizer`.
- Produces:
  - `class Store(Protocol)`: `put(text: str) -> str` (handle), `get(handle: str) -> str | None`. `MemoryStore()`; `DirStore(path)`. The handle is `"ctx_" + sha256(text)[:16]` in both, so equal content is stored once; `DirStore` writes `<path>/<handle>.txt` as UTF-8.
  - `Offloader(store: Store | None = None, *, max_tokens: int = 1500, preview_tokens: int = 200, adapter=OpenAIAdapter(), tokenizer=None)`; `offload(messages: list[M]) -> list[M]` is sync and returns a new list. A tool result over `max_tokens` becomes (exact copy): `"{preview}\n\n[truncated: {total_chars} characters stored as {handle}. Use ctx_read(handle, offset, limit) or ctx_grep(handle, pattern) to see more.]"`, where the preview is the first `preview_tokens * 4` characters.
  - `ctx_read(store, handle: str, offset: int = 0, limit: int = 2000) -> str` and `ctx_grep(store, handle: str, pattern: str, context: int = 2, max_matches: int = 20) -> str` (matching lines numbered `"{n}: {line}"`, blocks separated by `"--"`). Neither raises: an unknown handle → `"error: unknown handle {handle}"`; a bad regex → `"error: invalid pattern: {message}"`; no match → `"no matches"`. `Offloader.tools()` returns both bound to its store as plain functions with docstrings, for the integrations.

- [ ] **Step 1: Write failing tests**: `test_small_results_untouched`; `test_large_result_replaced_with_preview_and_handle`; `test_round_trip_read_full_content` (reading in 2000-char pages reassembles the original); `test_grep_context_and_limit`; `test_unknown_handle_and_bad_regex_return_errors`; `test_dirstore_dedupes_by_content` (tmp_path); `test_anthropic_tool_result_blocks`.
- [ ] **Step 2: Run** `uv run pytest tests/test_offloader.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `offloader.py`; export `Compactor`, `Offloader`, `MemoryStore`, `DirStore` from `astromesh_context.loop`.
- [ ] **Step 4: Run** the full suite and ruff. Expected: PASS.
- [ ] **Step 5: Commit** `feat: tool-result offloading with ctx_read and ctx_grep`.

---

### Task 12: ContextEval level 1 and the CLI

**Files:** Create `astromesh_context/eval/__init__.py`, `eval/dataset.py`, `eval/metrics.py`, `eval/runner.py`, `eval/data/nimbus/{corpus.jsonl,queries.jsonl,SOURCE.md}`, `astromesh_context/cli.py`, `examples/eval.yaml`; Modify `pyproject.toml` (include `eval/data` in the wheel), `.github/workflows/ci.yml` (eval smoke runs in the `rag`+`okf` extras job); Test `tests/test_eval_metrics.py`, `tests/test_eval_smoke.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: the public API, `RAGSource`, `OKFSource`, `OKFIndexSource`, `HistorySource`, `StaticSource`.
- Produces:
  - Vendored data: copy `../astromesh-rag/examples/corpus/corpus.jsonl` and `queries.jsonl` byte for byte; `SOURCE.md` records the source repo, commit (`git -C ../astromesh-rag rev-parse HEAD`) and Apache-2.0 license. `test_vendored_corpus_hash` pins both files' sha256.
  - `@dataclass(frozen=True) EvalCase(id, query, relevant: frozenset[str], type: str, history: list[dict], reference: str | None)`; `@dataclass EvalDataset(cases: list[EvalCase], docs: list[dict], distractors: frozenset[str], doc_id_of: Callable[[ContextItem], str | None])`.
  - `nimbus(history_turns: int = 4) -> EvalDataset`. Distractors = ids matching `pol-*-2024` plus docs whose `metadata.trap == "noise"`. History = `history_turns` deterministic user/assistant pairs built from a fixed list of eight generic Nimbus small-talk exchanges, rotated by case index. `doc_id_of`: `meta["doc_id"]` for rag items; the last path segment of `meta["concept_id"]` for okf items.
  - `nimbus_bundle(dest: Path) -> Path`: writes `policies/<doc_id>.md` for the 30 `pol-*` docs (frontmatter `type: Policy`, `title` = the first line of the text, `status: deprecated` for `-2024`, `verified` absent) and an `index.md`. Deterministic bytes.
  - Metrics (pure): `recall_at_budget(window_doc_ids: set, relevant: frozenset) -> float`; `precision_per_token(items, doc_id_of, relevant) -> float`; `distractor_rate(windows_doc_ids: list[set], distractors) -> float`; `prefix_reuse(hashes: list[str]) -> float` (share of consecutive pairs with equal hashes; 1.0 for fewer than 2).
  - `ContextEval(dataset: EvalDataset, configs: Mapping[str, Callable[[EvalDataset, int], Awaitable[ContextEngine]]], budgets: Sequence[int], *, level: int = 1, answer: Callable | None = None, judge: Callable | None = None)`; `async run() -> Report`. Sessions for `prefix_reuse` are consecutive groups of 5 cases sharing one growing history.
  - `Report` with `rows: list[dict]` (config, budget, recall, precision_per_token, distractor_rate, tokens_per_call, prefix_reuse, stable_tokens; plus accuracy, latency_ms and cost when level 2), `to_markdown() -> str`, `to_json() -> str` (also includes the corpus sha256, the configs and the command line).
  - `default_configs(embedder) -> dict`: builds `everything` (`Everything(truncate=True)`), `fixed_slots` (shares okf 0.3, rag 0.5, history 0.2), `scored_knapsack`, and, when `complete` is provided, `scored_knapsack+llm`, each over the same sources: system `StaticSource` (pinned, stable), `OKFIndexSource`, `OKFSource(limit=10)`, `RAGSource(k=20)` over an astromesh-rag hybrid pipeline on all 200 docs (one shared ingestion), and `HistorySource`.
  - CLI `astromesh-context eval [--level 1|2] [--budgets 2000,4000,8000] [--embedder hash|st] [--subset N] [--out DIR]` writes `DIR/<date>-<embedder>-L<level>.{json,md}` and prints the Markdown table; `astromesh-context explain --config policy.yaml --query "..."` prints `window.trace.explain()`.

- [ ] **Step 1: Write failing metric tests** with hand-computed values (for example recall of {a, b} against {a, c} = 0.5; `prefix_reuse(["x","x","y"]) == 0.5`).
- [ ] **Step 2: Write failing smoke tests**: `test_vendored_corpus_hash`; `test_nimbus_bundle_deterministic` (two writes give identical bytes); `test_eval_level1_subset` (`--subset 12`, `HashEmbedder`, budget 4000: four rows minus the llm config, all metrics in [0, 1]); `test_knapsack_not_below_fixed_slots_recall` (the regression floor from spec §4.3, on the subset); `test_cli_eval_writes_files` (tmp_path).
- [ ] **Step 3: Run** `uv run pytest tests/test_eval_metrics.py tests/test_eval_smoke.py tests/test_cli.py -q`. Expected: FAIL.
- [ ] **Step 4: Implement** the dataset, metrics, runner and CLI.
- [ ] **Step 5: Run** the full suite with extras and ruff. Expected: PASS. If `test_knapsack_not_below_fixed_slots_recall` fails, do not loosen it silently: report the numbers and stop for a decision.
- [ ] **Step 6: Commit** `feat: ContextEval level 1 on the Nimbus corpus and CLI`.

---

### Task 13: ContextEval level 2 (answers and judge)

**Files:** Create `astromesh_context/eval/judge.py`, `eval/data/nimbus/answers.jsonl`; Modify `eval/runner.py`, `eval/dataset.py` (load references), `cli.py`; Test `tests/test_eval_judge.py`

**Interfaces:**
- Consumes: `openai_compatible` (T7), `ContextEval` (T12).
- Produces:
  - `answers.jsonl`: one `{"id", "reference"}` per query id (60 lines). Each reference is a one-to-three-sentence answer written from the documents in `relevant` only, in the query's language. The implementer drafts them; **the user reviews them before any level-2 table is published** (record "reviewed by" in `SOURCE.md`).
  - `async judge_answer(complete, question, reference, answer) -> float` returns 1.0 / 0.5 / 0.0 from a judge reply of exactly `CORRECT`, `PARTIAL` or `WRONG` (anything else → 0.0 and a `judge_unparsed` count in the report). Judge prompt (exact copy): `"You grade an answer against a reference. Reply with exactly one word: CORRECT if the answer states the same facts as the reference, PARTIAL if it is incomplete but not wrong, WRONG otherwise.\n\nQuestion: {question}\nReference: {reference}\nAnswer: {answer}"`.
  - Level 2 answers with `window.to_messages()` sent to the answer model; it reports `accuracy`, p50 `latency_ms` of build + answer, and the token counts from the response `usage`. Environment: `CTX_EVAL_BASE_URL`, `CTX_EVAL_API_KEY`, `CTX_EVAL_MODEL`, `CTX_EVAL_JUDGE_MODEL`. The key is never printed or written; the report records only the host and the model ids.

- [ ] **Step 1: Write failing tests** with fake `complete` functions: `test_judge_maps_verdicts`; `test_judge_unparsed_is_zero_and_counted`; `test_answers_file_covers_every_query` (ids equal); `test_level2_row_has_accuracy` (fake answer and judge models); `test_api_key_not_in_report` (`respx` + a sentinel key string absent from `to_json()`).
- [ ] **Step 2: Run** `uv run pytest tests/test_eval_judge.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the judge and the level-2 path, and draft `answers.jsonl` by reading each relevant document in `corpus.jsonl`.
- [ ] **Step 4: Run** the full suite and ruff. Expected: PASS.
- [ ] **Step 5: Commit** `feat: ContextEval level 2 with reference answers and an LLM judge`. Then **stop and ask the user to review `answers.jsonl`.**

---

### Task 14: Integrations (LangChain, PydanticAI, MCP)

**Files:** Create `astromesh_context/integrations/__init__.py`, `_base.py`, `langchain.py`, `pydantic_ai.py`, `mcp.py`; Modify `cli.py` (`mcp` subcommand), `.github/workflows/ci.yml` (extras matrix); Test `tests/integrations/test_langchain.py`, `test_pydantic_ai.py`, `test_mcp.py`

**Interfaces:**
- Consumes: `ContextEngine`, `Compactor`, `Offloader`, `MessageAdapter` (T6, T10, T11).
- Produces:
  - `_base.require(module: str, extra: str)`, copied from astromesh-okf's helper: imports the module or raises `MissingExtraError(f"install astromesh-context[{extra}]")`.
  - `langchain.ContextMiddleware(engine: ContextEngine | None = None, *, compactor: Compactor | None = None, offloader: Offloader | None = None)`, an `AgentMiddleware`. Before each model call it offloads, then compacts the state messages (with a `LangChainAdapter` over `BaseMessage`, defined in this module), and when `engine` is set builds a window from the last human message and sets the request's system prompt to `window.to_text()`. `langchain.offload_tools(offloader) -> list` returns `ctx_read`/`ctx_grep` as LangChain tools. **First action of this step:** confirm the exact hook names (`before_model` / `wrap_model_call` and their async forms) and how the request's system prompt is replaced, against the installed `langchain` with the context7 docs; use what the installed version defines.
  - `pydantic_ai.history_processor(*, compactor=None, offloader=None)` returns an async callable `(list[ModelMessage]) -> list[ModelMessage]` using a `PydanticAIAdapter` (this module) over `ModelRequest`/`ModelResponse` parts (`ToolCallPart.tool_call_id`, `ToolReturnPart.tool_call_id`). `pydantic_ai.offload_tools(offloader)` returns `Tool` objects. The same confirm-against-installed-docs action applies.
  - `mcp.serve(engine, offloader=None, name="astromesh-context")` builds a `FastMCP` with tools `ctx_build(query: str) -> str` (returns `window.to_text()` followed by `trace.explain()`), `ctx_read` and `ctx_grep`. CLI: `astromesh-context mcp --config policy.yaml`.

- [ ] **Step 1: Write failing tests**, one per module, with each framework's fake model (`langchain_core.language_models.fake_chat_models.GenericFakeChatModel`; `pydantic_ai.models.test.TestModel`; MCP: call the tool functions registered on the FastMCP instance): the system prompt carries a known pinned string from the engine; a 5000-char tool result reaches the model offloaded; an over-threshold history reaches the model compacted with the tool pairs intact; `ctx_read` returns the stored content.
- [ ] **Step 2: Run** `uv sync --extra langchain --extra pydantic-ai --extra mcp` then `uv run pytest tests/integrations -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the three modules and the `mcp` CLI subcommand.
- [ ] **Step 4: Add** CI extras matrix entries for `langchain`, `pydantic-ai`, `mcp`; run `uv lock`.
- [ ] **Step 5: Run** the full suite with all extras and ruff. Expected: PASS.
- [ ] **Step 6: Commit** `feat: LangChain middleware, PydanticAI history processor and MCP server`.

---

### Task 15: Real eval run, README and docs

**Files:** Create `docs/extending.md`, `results/<date>-st-L1.{json,md}`, `results/<date>-st-L2.{json,md}`; Modify `README.md`, `README.es.md`, `CHANGELOG.md`

- [ ] **Step 1: Run level 1 with the real embedder**: `uv run --extra eval astromesh-context eval --level 1 --embedder st --out results/`. Expected: files written, four rows per budget.
- [ ] **Step 2: Run level 2** only after the user has reviewed `answers.jsonl` and provided the `CTX_EVAL_*` endpoint: `uv run --extra eval astromesh-context eval --level 2 --embedder st --out results/`.
- [ ] **Step 3: Update both READMEs**: remove the "in design" notice and badge; replace "Planned API" with the shipped API (test every snippet by running it); add an "Results" section with the Markdown table exactly as produced, the command, the models and the date, including any row where `scored_knapsack` does not win; `pip install` lines.
- [ ] **Step 4: Write `docs/extending.md`**: writing a `Source`, a `Policy`, a `Scorer`, and a `MessageAdapter`, each with a runnable example that is executed in `tests/test_docs_examples.py`.
- [ ] **Step 5: Run** the full suite and ruff. Expected: PASS.
- [ ] **Step 6: Commit** `docs: README with shipped API and first published eval results`.

---

### Task 16: Release 0.1.0 (each external step needs explicit user confirmation)

**Files:** Create `.github/workflows/release.yml` (copied from `../astromesh-okf/.github/workflows/release.yml`, names changed); Modify `CHANGELOG.md`

- [ ] **Step 1: Prerequisite check**: `pip index versions astromesh-rag` lists 0.1.x on PyPI. If not, stop: the `[rag]` and `[eval]` extras cannot install from PyPI. Ask the user whether to publish astromesh-rag first or release without those extras working from PyPI.
- [ ] **Step 2:** Move `[Unreleased]` to `## [0.1.0] - <date>` in `CHANGELOG.md`; commit `chore(release): astromesh-context 0.1.0`.
- [ ] **Step 3 (confirm with the user):** push `main`, tag `v0.1.0` (annotated), push the tag. The workflow publishes to TestPyPI, smoke-installs, then publishes to PyPI.
- [ ] **Step 4:** Verify `pip install astromesh-context==0.1.0` in a fresh venv and run the README quickstart.

### Task 17: Docs site (repo `../astromesh`, after 0.1.0 is on PyPI; confirm before pushing)

- [ ] Update `docs-site/src/data/ecosystem.ts` (`version: '0.1.0'`, real `released`, remove `inDevelopment`, `install: 'pip install astromesh-context'`, `latest` = one sentence on what shipped).
- [ ] Rewrite `context/introduction.mdx` without the caution aside; add `context/quickstart.mdx`, `context/policies.mdx`, `context/eval.mdx` (the published table), `context/integrations.mdx`; add them to the `Context` sidebar group in `astro.config.mjs`; update the version rows in `getting-started/ecosystem.md` and `architecture/ecosystem-dependencies.md`.
- [ ] `cd docs-site && npm run build`. Expected: build completes with no new warnings. Commit `docs(site): astromesh-context 0.1.0`.
