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
