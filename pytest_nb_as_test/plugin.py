"""pytest-nb-as-test plugin entry points."""

from __future__ import annotations

from importlib import metadata as importlib_metadata
from typing import Any, Generator

import pytest  # type: ignore
from packaging.version import InvalidVersion, Version

from .item import NotebookItem, pytest_collect_file  # pylint: disable=unused-import
from .options import pytest_addoption  # pylint: disable=unused-import


def _is_incompatible_pytest_asyncio_pair(
    *, pytest_version: str, pytest_asyncio_version: str
) -> bool:
    """Return whether version pair is known to break collection.

    Args:
        pytest_version: Installed pytest version string.
        pytest_asyncio_version: Installed pytest-asyncio version string.

    Returns:
        True when the pair is unsupported, else False.

    Example:
        _is_incompatible_pytest_asyncio_pair(
            pytest_version="9.0.2",
            pytest_asyncio_version="0.23.3",
        )
    """
    try:
        parsed_pytest_version: Version = Version(pytest_version)
        parsed_asyncio_version: Version = Version(pytest_asyncio_version)
    except InvalidVersion:
        return False
    return parsed_pytest_version >= Version(
        "9.0.0"
    ) and parsed_asyncio_version < Version("1.3.0")


def _raise_for_incompatible_pytest_asyncio(config: pytest.Config) -> None:
    """Fail early for known-bad pytest and pytest-asyncio pairings.

    Args:
        config: Pytest configuration object.

    Raises:
        pytest.UsageError: If installed plugin versions are known incompatible.

    Example:
        _raise_for_incompatible_pytest_asyncio(config)
    """
    if not config.pluginmanager.hasplugin("asyncio"):
        return

    try:
        pytest_asyncio_version: str = importlib_metadata.version("pytest-asyncio")
    except importlib_metadata.PackageNotFoundError:
        return

    if _is_incompatible_pytest_asyncio_pair(
        pytest_version=pytest.__version__,
        pytest_asyncio_version=pytest_asyncio_version,
    ):
        raise pytest.UsageError(
            "Unsupported plugin combination detected: "
            f"pytest=={pytest.__version__} with pytest-asyncio=={pytest_asyncio_version}. "
            "Use pytest-asyncio>=1.3.0, or pin pytest<9."
        )


def pytest_configure(config: pytest.Config) -> None:
    """Initialise the plugin and register the notebook marker.

    Args:
        config: Pytest configuration object.

    Example:
        pytest_configure(config)
    """
    _raise_for_incompatible_pytest_asyncio(config)

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
