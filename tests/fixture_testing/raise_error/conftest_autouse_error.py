"""Conftest that raises an error."""

import pytest


@pytest.fixture(autouse=True)
def explode() -> None:
    """This is a fixture that will just fail every time."""
    raise RuntimeError("boom")
