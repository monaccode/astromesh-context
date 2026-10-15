import sys

import pytest

from astromesh_context.errors import MissingExtraError
from astromesh_context.sources.okf import OKFIndexSource, OKFSource
from astromesh_context.sources.rag import RAGSource


def test_okf_missing_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "astromesh_okf", None)
    with pytest.raises(MissingExtraError, match=r"astromesh-context\[okf\]"):
        OKFSource("kb")
    with pytest.raises(MissingExtraError, match=r"astromesh-context\[okf\]"):
        OKFIndexSource("kb")


def test_rag_missing_extra_for_spec_path(monkeypatch):
    monkeypatch.setitem(sys.modules, "astromesh_rag", None)
    monkeypatch.setitem(sys.modules, "astromesh_rag.spec", None)
    with pytest.raises(MissingExtraError, match=r"astromesh-context\[rag\]"):
        RAGSource("rag.yaml")
