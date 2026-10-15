import pytest

from astromesh_context.errors import MissingExtraError
from astromesh_context.integrations._base import require


def test_require_imports_an_installed_module():
    assert require("json", "langchain").dumps([1]) == "[1]"


def test_require_names_the_extra_when_missing():
    with pytest.raises(MissingExtraError, match=r"install astromesh-context\[langchain\]"):
        require("astromesh_context_no_such_module", "langchain")
