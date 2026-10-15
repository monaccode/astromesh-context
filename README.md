<p align="center">
  <img src="https://raw.githubusercontent.com/monaccode/astromesh-context/main/assets/logo.png" alt="Astromesh Context" width="640">
</p>

# astromesh-context

**Decide what goes into your agent's context window, and prove it was the right call.**

[![astromesh-runtime](https://img.shields.io/badge/astromesh-runtime-f472b6)](https://monaccode.github.io/astromesh/#ecosystem)
[![Status: in design](https://img.shields.io/badge/status-in%20design-fb7185)](#status)
[![Python](https://img.shields.io/badge/python-3.12%2B-3776ab)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/monaccode/astromesh-context/blob/main/LICENSE)

*[Leer en español](https://github.com/monaccode/astromesh-context/blob/main/README.es.md)*

> **In design: nothing to install yet.** This README describes the design that v0.1 is being
> built against. Code examples show the planned API. The [roadmap](#roadmap) says what lands
> first, and this notice goes away when it does.

An agent's answer is only as good as what the model reads before it answers. Retrieval brings
candidates, a knowledge base brings curated concepts, memory brings what the agent learned,
and the conversation keeps growing. Something has to decide **which of all that fits in the
window, in what order, and what stays out**. Today that decision is scattered across prompt
templates, a `top_k` here and a history trim there, and nobody can tell you whether it was a
good one.

`astromesh-context` is that decision, made in one place:

- **One budget for every source.** RAG hits, OKF concepts, memory facts and chat history
  compete for the same tokens on a calibrated score, instead of each getting a fixed slice.
- **A trace for every item.** Everything a source proposed comes back marked included or
  excluded, with the reason: `budget`, `duplicate_of:…`, `below_floor`, `source_cap`,
  `deprecated`.
- **An order built for the prompt cache.** Stable content first, byte for byte, so
  consecutive calls share the longest possible prefix.
- **An eval that proves it.** `ContextEval` runs the same questions under different policies
  and reports accuracy, tokens spent and cache-prefix reuse side by side.

The core will depend only on PyYAML. [astromesh-rag](https://github.com/monaccode/astromesh-rag)
and [astromesh-okf](https://github.com/monaccode/astromesh-okf) plug in as sources through
optional extras.

## The map: six questions about context

Context and knowledge for agents is not one technique. It is six questions, and each has
its own patterns. This is where each Astromesh library stands.

| Question | Patterns | In Astromesh |
|---|---|---|
| **Which fragment helps?** (retrieval) | naive, hybrid, multi-query, HyDE, reranking, contextual retrieval, parent-document | [astromesh-rag](https://github.com/monaccode/astromesh-rag) |
| **Where do I go next?** (navigation) | index in the system prompt, agentic search with tools, llms.txt, progressive disclosure | [astromesh-okf](https://github.com/monaccode/astromesh-okf) |
| **How do things relate?** (graphs) | GraphRAG, entity knowledge graphs, temporal graphs | not yet |
| **What have I learned?** (memory) | conversational, semantic facts, episodic, self-managed memory, consolidation | Astromesh core; enters here as a source |
| **What fits in the window?** (context engineering) | token budgets across sources, compaction, tool-result offloading, cache-stable ordering, deduplication, lost-in-the-middle | **astromesh-context** |
| **Can I trust it?** (provenance) | citations, freshness, verification, conflicting sources | trust and freshness signals from okf, honored here |

The first two decide *what could be relevant*. This library decides *what the model actually
reads*.

## How a window is built

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#e0f2fe", "primaryTextColor": "#0a2463", "primaryBorderColor": "#1d4ed8", "lineColor": "#1d4ed8"}}}%%
flowchart TB
    Q([query]) --> S1[RAGSource]
    Q --> S2[OKFSource]
    Q --> S3[HistorySource]
    Q --> S4[your Source]
    S1 --> C[calibrate scores per source]
    S2 --> C
    S3 --> C
    S4 --> C
    C --> L[LLMScorer]:::llm
    L --> D[deduplicate]
    D --> P[policy: fill the budget]
    P --> O[order for the cache]
    O --> W([window and trace])
    classDef llm fill:#ede9fe,stroke:#7c3aed,stroke-dasharray: 5 4
```

Sources run in parallel and only **propose**: none of them knows the budget. Everything after
them is a pure function, so the same candidates and the same policy always give the same
window. The dashed step is optional and is the only one that calls a model.

1. **Calibrate.** A cosine of 0.82, a BM25 score of 14 and "the third most recent turn" are
   not comparable. Scores are normalized by rank within each source, then weighted by how much
   you trust that source. Signals adjust them where you can see it: a stale concept scores
   lower, a human-reviewed one higher, a deprecated one is left out.
2. **Score with a model (optional).** `LLMScorer` rescores the top candidates in one batched
   call and blends its score with the calibrated one. If it fails or times out, the window is
   built anyway on the calibrated scores and the trace says so.
3. **Deduplicate.** Exact copies by hash, near-copies by word shingles. This is what catches
   RAG and OKF returning the same policy twice.
4. **Fill the budget.** Pinned items (the system prompt, the OKF root index) go in first.
   Then a minimum per source (the last user turn, always), then the rest by value per token,
   respecting a cap per source and a score floor.
5. **Order for the cache.** Stable items first in a fixed order, then session items, then
   volatile ones, with the strongest at the edges of their block, where models read best.
   History stays chronological and last.

## Planned API

```python
from astromesh_context import ContextEngine, ScoredKnapsack, StaticSource, HistorySource
from astromesh_context.sources.rag import RAGSource
from astromesh_context.sources.okf import OKFIndexSource, OKFSource

engine = ContextEngine(
    sources=[
        StaticSource(system_prompt, kind="instruction", pinned=True),
        OKFIndexSource("./kb"),            # the bundle index, pinned in the stable prefix
        OKFSource("./kb", limit=10),       # whole concepts, with trust and freshness
        RAGSource(pipeline, k=20),         # chunks from an astromesh-rag pipeline
        HistorySource(messages),           # the conversation, newest turn always kept
    ],
    policy=ScoredKnapsack(),
    budget=8_000,
)

window = await engine.build("Can I get a refund after 3 weeks on the Team plan?")
messages = window.to_messages()            # OpenAI or Anthropic shape
print(window.trace.explain())              # what went in, what stayed out, and why
```

The same engine from YAML:

```yaml
apiVersion: astromesh/v1
kind: ContextPolicy
metadata:
  name: support
spec:
  budget: 8000
  policy: scored_knapsack
  sources:
    - { type: okf_index, bundle: ./kb, pinned: true }
    - { type: okf, bundle: ./kb, limit: 10, weight: 1.2 }
    - { type: rag, pipeline: support-docs, k: 20, max_tokens: 3000 }
    - { type: history, min_items: 2 }
```

## Policies

A policy decides how candidates fill the budget. They share one signature, so the eval
compares them as equals.

| Policy | What it does | Use it for |
|---|---|---|
| `ScoredKnapsack` | One budget, calibrated scores, value per token, caps and floors per source | The default |
| `FixedSlots` | A fixed share of the budget per source, filled in priority order | The baseline everyone runs today |
| `Everything` | Every candidate, or a clear error if they do not fit | The control: what "just stuff it all in" costs |

`LLMScorer` is not a policy: it is a scoring step that works with any of them.

## Inside the agent loop

Two optional pieces for long-running agents. They use the same types and do not need the
engine: each works on a plain list of OpenAI or Anthropic messages.

- **`Compactor`** summarizes old turns once history crosses a threshold and keeps the last
  turns verbatim. It never splits a tool call from its result, which providers reject. It is
  incremental: the previous summary feeds the next one.
- **`Offloader`** moves a tool result over the limit into a store and leaves a preview and a
  handle in its place. The agent reads what it needs with `ctx_read(handle, offset, limit)` and
  `ctx_grep(handle, pattern)`.

## Proving it: `ContextEval`

A context strategy is a claim about accuracy and cost, so it should come with numbers.
`ContextEval` runs a labeled question set through several configurations and reports, for
each one, answer accuracy, tokens per call, latency, and how much of the prompt prefix
consecutive calls share. The first published comparison will be the four configurations
above (`Everything`, `FixedSlots`, `ScoredKnapsack`, `ScoredKnapsack` with `LLMScorer`) on
the labeled corpus that ships with astromesh-rag.

## Integrations

v0.1 targets LangChain/LangGraph (a node before the model, and the loop pieces as a
`pre_model_hook`), PydanticAI (`history_processor`) and an MCP server exposing `ctx_build`,
`ctx_read` and `ctx_grep`. The other frameworks astromesh-okf supports follow.

## Roadmap

**v0.1:** `ContextEngine`; the three policies; calibration, deduplication and cache ordering;
`LLMScorer`; sources for RAG, OKF, history, static and callable content; `Compactor` and
`Offloader`; `ContextEval` with a first published comparison; the three integrations above;
`kind: ContextPolicy` YAML.

**Later:** more framework integrations; a graph source; adoption by the Astromesh core, where
it replaces today's history trimming and template-based prefetch.

## Status

In design. The repository holds the logo and this README; the written design for v0.1 is
being finished, and the code follows it. Follow the [Astromesh ecosystem](https://monaccode.github.io/astromesh/#ecosystem)
for the first release.

## License

Apache-2.0.
