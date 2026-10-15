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
        "---\ntype: Policy\ntitle: Política de reembolsos\nstatus: deprecated\n---\n"
    )
    assert new.startswith("---\ntype: Policy\ntitle: Política de reembolsos\n---\n")
    assert b"\r" not in (a / "index.md").read_bytes()


async def run_subset(budgets=(4000,), n=12):
    ds = nimbus().subset(n)
    configs = default_configs(HashEmbedder())
    return await ContextEval(ds, configs, list(budgets)).run()


async def test_eval_level1_subset():
    report = await run_subset()
    assert [r["config"] for r in report.rows] == ["everything", "fixed_slots", "scored_knapsack"]
    for row in report.rows:
        assert row["budget"] == 4000
        for key in ("recall", "precision_per_token", "distractor_rate", "prefix_reuse"):
            assert 0.0 <= row[key] <= 1.0, (row["config"], key)
        assert 0 < row["tokens_per_call"] <= 4000
        assert row["stable_tokens"] > 0
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
