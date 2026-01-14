"""Timeout handling for notebook execution."""

from __future__ import annotations

import time
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Iterator

import pytest  # type: ignore


@dataclass(frozen=True, kw_only=True)
class NotebookTimeoutConfig:  # pylint: disable=too-few-public-methods
    """Timeout configuration for notebook execution.

    Example:
        config = NotebookTimeoutConfig(
            notebook_timeout_seconds=60.0,
            default_cell_timeout_seconds=10.0,
        )
    """

    notebook_timeout_seconds: float | None
    """Maximum wall-clock time for an entire notebook, in seconds."""

    default_cell_timeout_seconds: float | None
    """Default per-cell timeout when no cell directive is provided, in seconds."""


class NotebookTimeoutController:  # pylint: disable=too-few-public-methods
    """Manage per-cell timeouts using pytest-timeout hooks.

    Example:
        controller = NotebookTimeoutController(item, timeout_config, True)
        with controller.cell_timeout_context(None, cell_index=1):
            ...  # run cell body
    """

    def __init__(
        self,
        item: pytest.Item,
        timeout_config: NotebookTimeoutConfig,
        has_timeouts: bool,
    ) -> None:
        self._item = item
        self._timeout_config = timeout_config
        self._has_timeouts = has_timeouts
        self._notebook_start_s = time.monotonic()
        self._settings = None
        if self._has_timeouts:
            self._settings = _get_pytest_timeout_settings(item.config)

    def cell_timeout_context(
        self,
        cell_timeout_seconds: float | None,
        cell_index: int,
    ) -> AbstractContextManager[None]:
        """Return a context manager that enforces the effective cell timeout.

        Args:
            cell_timeout_seconds: Optional per-cell timeout override in seconds.
            cell_index: Index of the cell for error messages.

        Returns:
            A context manager that sets a pytest-timeout timer when needed.

        Example:
            with controller.cell_timeout_context(0.5, cell_index=3):
                ...
        """
        effective_timeout = self._effective_timeout(cell_timeout_seconds, cell_index)
        if effective_timeout is None:
            return nullcontext()
        if self._settings is None:
            raise pytest.UsageError(
                "Notebook timeouts require pytest-timeout to be installed."
            )
        return _pytest_timeout_context(
            item=self._item,
            settings=self._settings,
            timeout_seconds=effective_timeout,
        )

    def _effective_timeout(
        self,
        cell_timeout_seconds: float | None,
        cell_index: int,
    ) -> float | None:
        """Compute the effective timeout for a cell.

        Args:
            cell_timeout_seconds: Optional per-cell timeout override in seconds.
            cell_index: Index of the cell for error messages.

        Returns:
            The effective timeout in seconds, or None if no timeout applies.

        Example:
            timeout = controller._effective_timeout(None, cell_index=0)
        """
        if not self._has_timeouts:
            return None
        candidate_timeouts: list[float] = []
        if cell_timeout_seconds is not None:
            candidate_timeouts.append(cell_timeout_seconds)
        default_cell = self._timeout_config.default_cell_timeout_seconds
        if default_cell is not None:
            candidate_timeouts.append(default_cell)
        notebook_timeout = self._timeout_config.notebook_timeout_seconds
        if notebook_timeout is not None:
            elapsed_s = time.monotonic() - self._notebook_start_s
            remaining_s = notebook_timeout - elapsed_s
            if remaining_s <= 0:
                pytest.fail(
                    f"Notebook timeout ({notebook_timeout:.3f}s) exceeded before cell "
                    f"{cell_index}."
                )
            candidate_timeouts.append(remaining_s)
        if not candidate_timeouts:
            return None
        return min(candidate_timeouts)


def _has_pytest_timeout_hooks(config: pytest.Config) -> bool:
    """Return True if pytest-timeout hooks are available.

    Args:
        config: Pytest configuration object.

    Returns:
        True when pytest-timeout hooks are registered.

    Example:
        if _has_pytest_timeout_hooks(config):
            ...
    """
    hooks = config.pluginmanager.hook
    return hasattr(hooks, "pytest_timeout_set_timer") and hasattr(
        hooks, "pytest_timeout_cancel_timer"
    )


def _get_pytest_timeout_settings(config: pytest.Config) -> Any:
    """Load pytest-timeout settings for the current pytest config.

    Args:
        config: Pytest configuration object.

    Returns:
        pytest-timeout settings namedtuple.

    Example:
        settings = _get_pytest_timeout_settings(config)
    """
    import pytest_timeout  # type: ignore  # pylint: disable=import-outside-toplevel

    return pytest_timeout.get_env_settings(config)


@contextmanager
def _pytest_timeout_context(
    item: pytest.Item,
    settings: Any,
    timeout_seconds: float,
) -> Iterator[None]:
    """Context manager that arms pytest-timeout for a block.

    Args:
        item: Pytest item used by pytest-timeout hooks.
        settings: pytest-timeout settings namedtuple.
        timeout_seconds: Timeout duration in seconds.

    Yields:
        None.

    Example:
        with _pytest_timeout_context(item, settings, 0.5):
            ...
    """
    hooks = item.config.pluginmanager.hook
    updated_settings = settings._replace(timeout=timeout_seconds)
    hooks.pytest_timeout_set_timer(item=item, settings=updated_settings)
    try:
        yield
    finally:
        hooks.pytest_timeout_cancel_timer(item=item)
