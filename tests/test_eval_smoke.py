import dataclasses
import hashlib
from importlib import resources

import pytest

pytest.importorskip("astromesh_rag")
pytest.importorskip("astromesh_okf")

from astromesh_rag.testing import HashEmbedder

from astromesh_context.eval.dataset import nimbus, nimbus_bundle
from astromesh_context.eval.runner import ContextEval, default_configs

DATA = resources.files("astromesh_context.eval") / "data" / "nimbus"
PINNED = {
    "corpus.jsonl": "44d13cfb3a0f6eeac8e25c54c628b809e56b205355f4071c09a1942a198078da",
    "queries.jsonl": "3619ce0b66c0ed5736988aa6be333378b456c8acc5221e7c5ad50ff4c5043d4c",
}


def test_vendored_corpus_hash():
    for name, digest in PINNED.items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest, name
    assert "a86bfba" in (DATA / "SOURCE.md").read_text(encoding="utf-8")


def test_nimbus_shape():
    ds = nimbus()
    assert len(ds.docs) == 200
    assert len(ds.cases) == 60
    assert "pol-reembolsos-2024" in ds.distractors
    assert "pol-reembolsos-2026" not in ds.distractors
    assert len(ds.distractors) == 15 + 60
    assert len(ds.superseded) == 15
    assert len(ds.noise) == 60
    assert ds.distractors == ds.superseded | ds.noise
    assert all(len(c.history) == 8 for c in ds.cases)
    assert ds.cases[0].history != ds.cases[1].history  # rotated by case index


def test_nimbus_bundle_deterministic(tmp_path):
    a = nimbus_bundle(tmp_path / "a")
    b = nimbus_bundle(tmp_path / "b")
    files_a = sorted(p.relative_to(a).as_posix() for p in a.rglob("*") if p.is_file())
    files_b = sorted(p.relative_to(b).as_posix() for p in b.rglob("*") if p.is_file())
    assert files_a == files_b
    assert len([f for f in files_a if f.startswith("policies/")]) == 30
    assert "index.md" in files_a
    assert all((a / f).read_bytes() == (b / f).read_bytes() for f in files_a)
    old = (a / "policies" / "pol-reembolsos-2024.md").read_bytes().decode("utf-8")
    new = (a / "policies" / "pol-reembolsos-2026.md").read_bytes().decode("utf-8")
    assert old.startswith(
        "---\ntype: Policy\ntitle: Política de reembolsos\nstatus: deprecated\n"
        "superseded_by: pol-reembolsos-2026\n---\n"
    )
    assert old.endswith("\n\nReemplazada por [pol-reembolsos-2026](pol-reembolsos-2026.md).\n")
    assert "superseded_by" not in new
    assert "Reemplazada por [" not in new
    assert new.startswith("---\ntype: Policy\ntitle: Política de reembolsos\n---\n")
    assert b"\r" not in (a / "index.md").read_bytes()


async def run_subset(budgets=(1000,), n=12):
    ds = nimbus().subset(n)
    configs = default_configs(HashEmbedder())
    return await ContextEval(ds, configs, list(budgets)).run()


async def test_eval_level1_subset():
    report = await run_subset()
    assert [r["config"] for r in report.rows] == ["everything", "fixed_slots", "scored_knapsack"]
    for row in report.rows:
        assert row["budget"] == 1000
        for key in ("recall", "precision_per_token", "distractor_rate", "prefix_reuse"):
            assert 0.0 <= row[key] <= 1.0, (row["config"], key)
        assert 0 < row["tokens_per_call"] <= 1000
        assert row["stable_tokens"] > 0
        assert row["candidate_tokens"] >= row["tokens_per_call"]
        for key in ("superseded_rate", "noise_share"):
            assert 0.0 <= row[key] <= 1.0, (row["config"], key)
    assert "candidate tokens" in report.to_markdown()
    assert "| scored_knapsack |" in report.to_markdown()
    assert '"corpus_sha256"' in report.to_json()


async def test_knapsack_not_below_fixed_slots_recall():
    report = await run_subset()
    recall = {r["config"]: r["recall"] for r in report.rows}
    assert recall["scored_knapsack"] >= recall["fixed_slots"], recall


def test_unsupported_level():
    ds = dataclasses.replace(nimbus(), cases=[])
    with pytest.raises(ValueError, match="level"):
        ContextEval(ds, {}, [4000], level=3)


async def test_report_json_has_config_meta_and_no_command():
    import json

    report = await run_subset(n=5)
    data = json.loads(report.to_json())
    assert data["command"] is None
    cm = data["config_meta"]
    assert cm["fixed_slots"]["policy"] == "FixedSlots"
    assert cm["fixed_slots"]["shares"] == {"okf": 0.3, "rag": 0.5, "history": 0.2}
    assert cm["scored_knapsack"]["policy"] == "ScoredKnapsack"
    assert cm["everything"]["truncate"] is True
    for c in cm.values():
        assert c["rag_k"] == 20
        assert c["okf_limit"] == 10
        assert c["splitter"] == {"size": 800, "overlap": 80}
        assert c["embedder"] == "HashEmbedder"
    assert data["history_turns"] == 4


async def test_default_configs_with_complete_adds_llm_config():
    from astromesh_context.scorer import LLMScorer

    async def complete(prompt):
        return "[]"

    configs = default_configs(HashEmbedder(), complete=complete)
    assert list(configs) == ["everything", "fixed_slots", "scored_knapsack", "scored_knapsack+llm"]
    engine = await configs["scored_knapsack+llm"](nimbus().subset(1), 1000)
    assert isinstance(engine.scorer, LLMScorer)
    plain = await configs["scored_knapsack"](nimbus().subset(1), 1000)
    assert plain.scorer is None
    assert configs["scored_knapsack+llm"].meta["scorer"] == "LLMScorer"
