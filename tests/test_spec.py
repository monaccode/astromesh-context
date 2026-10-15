from pathlib import Path

import pytest

from astromesh_context import (
    ContextEngine,
    Everything,
    FixedSlots,
    HistorySource,
    LLMScorer,
    ScoredKnapsack,
    SpecError,
    load_engine,
)
from astromesh_context.spec import load_engine_dict

FIXTURES = Path(__file__).parent / "fixtures"


def _raw(sources=None, **spec):
    body = {"budget": 1000, "sources": sources or [{"type": "static", "text": "hi"}]}
    body.update(spec)
    return {"apiVersion": "astromesh/v1", "kind": "ContextPolicy", "spec": body}


def _load(raw, tmp_path, **kw):
    return load_engine_dict(raw, base_dir=tmp_path, **kw)


def test_loads_full_fixture():
    pytest.importorskip("astromesh_okf")
    engine = load_engine(FIXTURES / "policy.yaml")
    assert isinstance(engine, ContextEngine)
    assert [c.source.name for c in engine.configs] == ["system", "index", "kb", "history"]
    assert engine.budget_total == 4000
    assert isinstance(engine.policy, ScoredKnapsack)
    assert engine.policy.beta == 0.5
    kb = engine.configs[2]
    assert kb.weight == 1.5
    assert kb.max_tokens == 1500
    assert engine.configs[3].min_items == 2


def test_policy_string_shorthand(tmp_path):
    assert isinstance(_load(_raw(policy="everything"), tmp_path).policy, Everything)
    fs = _load(_raw(policy={"type": "fixed_slots", "shares": {"a": 0.5}}), tmp_path).policy
    assert isinstance(fs, FixedSlots)
    assert fs.shares == {"a": 0.5}


def test_unknown_key_reports_path(tmp_path):
    raw = _raw([{"type": "static", "text": "x"}, {"type": "history", "lmit": 3}])
    with pytest.raises(SpecError) as e:
        _load(raw, tmp_path)
    assert e.value.path == "spec.sources[1].lmit"


def test_wrong_kind_rejected(tmp_path):
    raw = _raw()
    raw["kind"] = "Agent"
    with pytest.raises(SpecError) as e:
        _load(raw, tmp_path)
    assert e.value.path == "kind"


def test_scorer_without_complete_rejected(tmp_path):
    with pytest.raises(SpecError) as e:
        _load(_raw(scorer={"type": "llm", "top_n": 5}), tmp_path)
    assert e.value.path == "spec.scorer"

    async def complete(prompt: str) -> str:
        return "[]"

    engine = _load(_raw(scorer="llm"), tmp_path, complete=complete)
    assert isinstance(engine.scorer, LLMScorer)


def test_text_file_relative_to_yaml(tmp_path):
    (tmp_path / "p.md").write_text("from file", encoding="utf-8")
    raw = _raw([{"type": "static", "text_file": "p.md"}])
    engine = _load(raw, tmp_path)
    assert engine.configs[0].source._item.content == "from file"


def test_object_reference_missing_rejected(tmp_path):
    raw = _raw([{"type": "callable", "name": "c", "object": "fn"}])
    with pytest.raises(SpecError) as e:
        _load(raw, tmp_path)
    assert e.value.path == "spec.sources[0].object"
    engine = _load(raw, tmp_path, objects={"fn": lambda q, c: []})
    assert engine.configs[0].source.name == "c"


def test_yaml_root_not_mapping(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("- a\n", encoding="utf-8")
    with pytest.raises(SpecError) as e:
        load_engine(p)
    assert e.value.path == ""


def test_bare_history_keeps_engine_default(tmp_path):
    engine = _load(_raw([{"type": "history"}]), tmp_path)
    assert isinstance(engine.configs[0].source, HistorySource)
    assert engine.configs[0].min_items == 2


@pytest.mark.parametrize(
    ("override", "path"),
    [
        ({"budget": 0}, "spec.budget"),
        ({"budget": True}, "spec.budget"),
        ({"policy": {"type": "everything", "bogus": 1}}, "spec.policy.bogus"),
        ({"policy": "nope"}, "spec.policy"),
        ({"tokenizer": "words"}, "spec.tokenizer"),
        ({"sources": [{"type": "history", "pinned": True}]}, "spec.sources[0].pinned"),
        ({"sources": [{"type": "history", "weight": "x"}]}, "spec.sources[0].weight"),
        ({"sources": [{"type": "history", "timeout": 0}]}, "spec.sources[0].timeout"),
        ({"sources": [{"type": "history", "mode": "bad"}]}, "spec.sources[0].mode"),
        ({"sources": [{"type": "history", "max_tokens": -1}]}, "spec.sources[0].max_tokens"),
        ({"sources": [{"type": "nope"}]}, "spec.sources[0].type"),
        ({"sources": [{"type": "static"}]}, "spec.sources[0].text"),
        ({"bogus": 1}, "spec.bogus"),
    ],
)
def test_bad_values_report_path(tmp_path, override, path):
    raw = _raw()
    raw["spec"].update(override)
    with pytest.raises(SpecError) as e:
        _load(raw, tmp_path)
    assert e.value.path == path


def test_spec_example_loads_verbatim(tmp_path, monkeypatch):
    pytest.importorskip("astromesh_okf")
    import shutil

    from astromesh_context.sources import rag as rag_mod

    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "system.md").write_text("system", encoding="utf-8")
    shutil.copytree(FIXTURES / "kb", tmp_path / "kb")
    seen = []

    def fake_load(path):
        seen.append(path)
        return object()

    monkeypatch.setattr(rag_mod, "_load", fake_load)

    async def complete(prompt: str) -> str:
        return "[]"

    (tmp_path / "policy.yaml").write_text(
        """\
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
""",
        encoding="utf-8",
    )
    engine = load_engine(tmp_path / "policy.yaml", complete=complete)
    assert [c.source.name for c in engine.configs] == [
        "static",
        "okf_index",
        "okf",
        "rag",
        "history",
    ]
    assert engine.configs[3].max_tokens == 3000
    assert seen == [tmp_path / "pipelines" / "support.rag.yaml"]


def test_okf_index_pinned_false_rejected(tmp_path):
    pytest.importorskip("astromesh_okf")
    raw = _raw([{"type": "okf_index", "bundle": str(FIXTURES / "kb"), "pinned": False}])
    with pytest.raises(SpecError) as e:
        _load(raw, tmp_path)
    assert e.value.path == "spec.sources[0].pinned"


def test_rag_k_forwarded(tmp_path):
    pipe = object()
    raw = _raw([{"type": "rag", "pipeline": {"object": "p"}, "k": 7}])
    engine = _load(raw, tmp_path, objects={"p": pipe})
    assert engine.configs[0].source._k == 7
    raw["spec"]["sources"][0]["k"] = 0
    with pytest.raises(SpecError) as e:
        _load(raw, tmp_path, objects={"p": pipe})
    assert e.value.path == "spec.sources[0].k"


@pytest.mark.parametrize("key", ["kind", "stability"])
def test_bad_kind_rejected(tmp_path, key):
    raw = _raw([{"type": "static", "text": "x", key: "bogus"}])
    with pytest.raises(SpecError) as e:
        _load(raw, tmp_path)
    assert e.value.path == f"spec.sources[0].{key}"


def test_missing_file_is_spec_error(tmp_path):
    missing = tmp_path / "nope.yaml"
    with pytest.raises(SpecError) as e:
        load_engine(missing)
    assert e.value.path == ""
    assert "nope.yaml" in str(e.value)


def test_invalid_yaml_is_spec_error(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("a: [unclosed\n", encoding="utf-8")
    with pytest.raises(SpecError) as e:
        load_engine(p)
    assert e.value.path == ""
