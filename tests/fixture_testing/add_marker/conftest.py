"""Fixtures for notebook marker environment setup."""

from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def set_env_for_notebooks(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Set the notebook fixture env var only for notebook-marked tests.

    Args:
        request: Pytest fixture request for marker inspection.
        monkeypatch: Pytest monkeypatch helper for env var updates.

    Example:
        pytest -k notebook
    """
    if request.node.get_closest_marker("notebook") is None:
        yield
        return
    monkeypatch.setenv("PYTEST_NOTEBOOK_FIXTURE", "1")
    yield
