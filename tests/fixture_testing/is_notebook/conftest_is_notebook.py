"""Notebook-only matplotlib backend configuration."""

from typing import Iterator

import matplotlib
import pytest


@pytest.fixture(autouse=True)
def set_matplotlib_backend(request: pytest.FixtureRequest) -> Iterator[None]:
    """Switch matplotlib backend for notebook-marked tests only.

    Args:
        request: Pytest fixture request for marker inspection.

    Example:
        pytest -k test_conftest_notebook_detection_sets_matplotlib_backend
    """

    if request.node.get_closest_marker("notebook") is None:
        matplotlib.use("SVG")
        yield
        return

    matplotlib.use("Agg")
    yield
