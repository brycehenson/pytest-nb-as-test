"""Tests for non-notebook matplotlib backend handling."""

import matplotlib


def test_regular_backend_unchanged() -> None:
    """Ensure the backend stays unset for non-notebook tests.

    Example:
        pytest -k regular_backend
    """
    backend = matplotlib.get_backend()
    assert backend == "SVG"
