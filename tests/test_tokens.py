import sys

import pytest

from astromesh_context.errors import MissingExtraError
from astromesh_context.tokens import CharEstimator, TiktokenTokenizer


def test_char_estimator():
    assert CharEstimator().count("abcde") == 2
    assert CharEstimator().count("") == 0


def test_tiktoken_missing_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    with pytest.raises(MissingExtraError):
        TiktokenTokenizer().count("x")
