"""pytest-nb-as-test plugin entry points."""

from __future__ import annotations

from typing import Any, Generator

import pytest  # type: ignore

from .item import NotebookItem, pytest_collect_file  # pylint: disable=unused-import
from .options import pytest_addoption  # pylint: disable=unused-import


def pytest_configure(config: pytest.Config) -> None:
    """Initialise the plugin and register the notebook marker.

    Args:
        config: Pytest configuration object.

    Example:
        pytest_configure(config)
    """
    # register a custom marker so that users can select notebook tests
    config.addinivalue_line(
        "markers", "notebook: mark test as generated from a Jupyter notebook"
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo
) -> Generator[None, Any, Any]:
    """Attach generated code to reports when requested.

    This hook is called for every phase of a test run. When the item is a
    NotebookItem it calls ``_dump_generated_code`` to attach the source
    code to the report if configured to do so. The hook is implemented
    as a wrapper using ``yield`` to access the generated report.

    Args:
        item: Pytest item being executed.
        call: Pytest call info for the current test phase.

    Yields:
        None.

    Example:
        outcome = pytest_runtest_makereport(item, call)
    """
    # The hook spec requires the argument name "call".
    # pylint: disable=unused-argument
    outcome = yield
    rep = outcome.get_result()
    if isinstance(item, NotebookItem):
        # rep is a TestReport for call and for setup/teardown phases
        item._dump_generated_code(rep)  # pylint: disable=protected-access
