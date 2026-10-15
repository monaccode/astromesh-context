import json
import re

import pytest

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
    assert len(data["rows"]) == 3
    assert data["command"].startswith("astromesh-context eval --subset 6")
    assert "scored_knapsack" in capsys.readouterr().out


def test_cli_rejects_level_2(tmp_path):
    with pytest.raises(SystemExit):
        main(["eval", "--level", "2", "--out", str(tmp_path)])


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
    with pytest.raises(MissingExtraError, match=r"\[rag\]"):
        _embedder("hash")
    monkeypatch.setitem(sys.modules, "astromesh_rag.embedders", None)
    with pytest.raises(MissingExtraError, match=r"\[eval\]"):
        _embedder("st")
