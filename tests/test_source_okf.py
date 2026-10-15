import asyncio
from pathlib import Path

import pytest

pytest.importorskip("astromesh_okf")

from astromesh_context.sources.okf import OKFIndexSource, OKFSource
from astromesh_context.types import Budget, BuildContext

KB = Path(__file__).parent / "fixtures" / "kb"


def make_ctx():
    return BuildContext(query="q", budget=Budget(total=100), deadline=0.0, vars={})


async def search(query, bundle=KB, **kw):
    return await OKFSource(bundle, **kw).candidates(query, make_ctx())


async def test_index_source_pinned_stable():
    (item,) = OKFIndexSource(KB).candidates("anything", make_ctx())
    assert item.id == "okf_index:/"
    assert (item.kind, item.stability, item.pinned) == ("instruction", "stable", True)
    assert item.source == "okf_index"
    assert "policies" in item.content


async def test_okf_ranks_by_terms():
    items = await search("refund days")
    assert items[0].id == "okf:policies/refunds"
    assert items[0].content.startswith("# Refund policy\n\n")
    assert items[0].meta == {"concept_id": "policies/refunds"}
    assert items[0].signals.citation == "policies/refunds"
    assert (items[0].kind, items[0].stability) == ("knowledge", "volatile")
    assert all(i.score > 0 for i in items)
    assert [i.score for i in items] == sorted((i.score for i in items), reverse=True)


async def test_okf_signals_mapped():
    by_id = {i.id: i for i in await search("refund shipping policy")}
    refunds = by_id["okf:policies/refunds"].signals
    assert refunds.trust == "human-reviewed"
    assert not refunds.stale
    assert not refunds.deprecated
    assert by_id["okf:policies/refunds-v1"].signals.deprecated is True
    shipping = by_id["okf:policies/shipping"].signals
    assert shipping.stale is True
    assert shipping.trust == "unverified"


async def test_okf_skips_reserved_and_invalid(tmp_path):
    (tmp_path / "good.md").write_text("---\ntype: note\ntitle: Good\n---\nwidget\n", "utf-8")
    (tmp_path / "bad.md").write_text("widget without frontmatter\n", "utf-8")
    (tmp_path / "index.md").write_text("widget widget widget\n", "utf-8")
    (tmp_path / "log.md").write_text("widget widget widget\n", "utf-8")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "h.md").write_text("---\ntype: note\n---\nwidget\n", "utf-8")
    items = await search("widget", bundle=tmp_path)
    assert [i.id for i in items] == ["okf:good"]


async def test_okf_accent_insensitive_case():
    items = await search("DEVOLUCIÓN")
    assert items[0].id == "okf:policies/devoluciones"


async def test_okf_empty_query_and_limit():
    assert await search("") == []
    assert await search("   \n") == []
    assert await search("zzzunknown") == []
    assert len(await search("policy", limit=2)) == 2


async def test_okf_runs_off_the_event_loop_thread():
    # Scoring happens in a worker thread; the loop stays responsive meanwhile.
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0)

    task = asyncio.create_task(ticker())
    await search("refund")
    task.cancel()
    assert ticks >= 1
