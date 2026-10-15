import json
import re

import httpx
import pytest
import respx

pytest.importorskip("astromesh_rag")
pytest.importorskip("astromesh_okf")

from astromesh_context.cli import main


def test_cli_eval_writes_files(tmp_path, capsys):
    code = main(
        ["eval", "--subset", "6", "--budgets", "4000", "--embedder", "hash", "--out", str(tmp_path)]
    )
    assert code == 0
    names = sorted(p.name for p in tmp_path.iterdir())
    assert len(names) == 2
    assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}-hash-L1\.(json|md)", n) for n in names)
    data = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert len(data["rows"]) == 4
    assert data["command"].startswith("astromesh-context eval --subset 6")
    assert "scored_knapsack" in capsys.readouterr().out


def test_cli_rejects_level_3(tmp_path):
    with pytest.raises(SystemExit):
        main(["eval", "--level", "3", "--out", str(tmp_path)])


def test_cli_explain(tmp_path, capsys):
    cfg = tmp_path / "p.yaml"
    cfg.write_text(
        "apiVersion: astromesh/v1\nkind: ContextPolicy\nspec:\n  budget: 500\n  sources:\n"
        "    - { type: static, text: 'You are Nimbus support.', pinned: true }\n",
        encoding="utf-8",
    )
    assert main(["explain", "--config", str(cfg), "--query", "hola"]) == 0
    out = capsys.readouterr().out
    assert "static:" in out
    assert "reason" in out


def test_cli_mcp_serves_the_policy(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from astromesh_context.integrations import mcp as mcp_mod

    cfg = tmp_path / "p.yaml"
    cfg.write_text(
        "apiVersion: astromesh/v1\nkind: ContextPolicy\nspec:\n  budget: 500\n  sources:\n"
        "    - { type: static, text: 'You are Nimbus support.', pinned: true }\n",
        encoding="utf-8",
    )
    ran = []

    class FakeServer:
        def run(self):
            ran.append(True)

    seen = {}

    def fake_serve(engine, offloader=None, name="astromesh-context"):
        seen["budget"], seen["name"] = engine.budget_total, name
        return FakeServer()

    monkeypatch.setattr(mcp_mod, "serve", fake_serve)
    assert main(["mcp", "--config", str(cfg)]) == 0
    assert ran == [True]
    assert seen == {"budget": 500, "name": "astromesh-context"}


def test_cli_explain_bad_config(tmp_path, capsys):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("kind: nope\n", encoding="utf-8")
    assert main(["explain", "--config", str(cfg), "--query", "q"]) == 2
    assert "error" in capsys.readouterr().err.lower()


def test_cli_default_budgets():
    from astromesh_context.cli import build_parser

    assert build_parser().parse_args(["eval"]).budgets == [750, 1000, 2000]


def test_cli_explain_missing_file(tmp_path, capsys):
    assert main(["explain", "--config", str(tmp_path / "nope.yaml"), "--query", "q"]) == 2
    assert capsys.readouterr().err.startswith("error:")


def test_cli_eval_unwritable_out(tmp_path, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    code = main(["eval", "--subset", "1", "--budgets", "1000", "--out", str(blocker / "sub")])
    assert code == 2
    assert capsys.readouterr().err.startswith("error:")


def test_missing_extra_names_the_right_extra(monkeypatch):
    import sys

    from astromesh_context.cli import _embedder
    from astromesh_context.errors import MissingExtraError

    monkeypatch.setitem(sys.modules, "astromesh_rag.testing", None)
    with pytest.raises(MissingExtraError, match=r"\[rag,okf\]"):
        _embedder("hash")
    monkeypatch.setitem(sys.modules, "astromesh_rag.embedders", None)
    with pytest.raises(MissingExtraError, match=r"\[eval\]"):
        _embedder("st")


@respx.mock
def test_cli_level2_runs_the_llm_scorer_row(tmp_path, monkeypatch):
    monkeypatch.setenv("CTX_EVAL_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("CTX_EVAL_API_KEY", "sk-test")
    monkeypatch.setenv("CTX_EVAL_MODEL", "answer-m")
    monkeypatch.delenv("CTX_EVAL_JUDGE_MODEL", raising=False)
    reply = {"choices": [{"message": {"content": "CORRECT"}}], "usage": {}}
    respx.post("https://llm.example.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=reply)
    )
    code = main(
        ["eval", "--level", "2", "--subset", "2", "--budgets", "1000", "--out", str(tmp_path)]
    )
    assert code == 0
    data = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert "scored_knapsack+llm" in {row["config"] for row in data["rows"]}
