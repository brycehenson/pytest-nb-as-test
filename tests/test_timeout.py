"""Unit tests for timeout integration helpers."""

from __future__ import annotations

from collections import namedtuple
from types import SimpleNamespace
from typing import Any

from pytest_nb_as_test.timeout import _pytest_timeout_context


def test_pytest_timeout_context_restores_outer_cancel_handle() -> None:
    """Preserve an outer timeout cancel callback across nested timeout usage."""
    outer_cancel_count = 0
    inner_cancel_count = 0

    class _HookProxy:
        """Minimal hook proxy that mirrors pytest-timeout cancel behavior."""

        def pytest_timeout_set_timer(self, *, item: Any, settings: Any) -> None:
            """Set a synthetic timer by installing a cancel callback on the item."""
            del settings

            def _cancel_inner() -> None:
                nonlocal inner_cancel_count
                inner_cancel_count += 1

            item.cancel_timeout = _cancel_inner

        def pytest_timeout_cancel_timer(self, *, item: Any) -> None:
            """Invoke the currently installed cancel callback for the item."""
            cancel = getattr(item, "cancel_timeout", None)
            if cancel is not None:
                cancel()

    hook_proxy = _HookProxy()
    item = SimpleNamespace(
        config=SimpleNamespace(pluginmanager=SimpleNamespace(hook=hook_proxy))
    )

    def _cancel_outer() -> None:
        nonlocal outer_cancel_count
        outer_cancel_count += 1

    item.cancel_timeout = _cancel_outer
    settings = namedtuple("Settings", ["timeout"])(timeout=30.0)

    with _pytest_timeout_context(item=item, settings=settings, timeout_seconds=0.5):
        pass

    assert inner_cancel_count == 1
    assert outer_cancel_count == 0
    assert item.cancel_timeout is _cancel_outer

    hook_proxy.pytest_timeout_cancel_timer(item=item)
    assert outer_cancel_count == 1
