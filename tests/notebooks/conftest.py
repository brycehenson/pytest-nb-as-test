"""Notebook-local pytest configuration for runtime setup/teardown."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def configure_notebook_runtime(request: pytest.FixtureRequest) -> None:
    """Configure optional runtime settings for notebook tests.

    Args:
        request: Pytest fixture request for the current test node.

    Returns:
        None. Uses a yield fixture to allow post-notebook teardown.

    Example:
        Place this file at tests/notebooks/conftest.py to apply only to
        notebooks under that directory.
    """
    if request.node.get_closest_marker("notebook") is None:
        yield
        return

    try:
        import numpy as np
    except ModuleNotFoundError:
        np = None
    if np is not None:
        np.random.seed(42)

    try:
        import matplotlib
    except ModuleNotFoundError:
        matplotlib = None
    if matplotlib is not None:
        matplotlib.use("Agg")

    try:
        import plotly.io as pio
    except ModuleNotFoundError:
        pio = None
    if pio is not None:
        pio.renderers.default = "jpg"

    yield
