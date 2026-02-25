"""Multiprocessing safety guards for notebook execution.

This module provides a scoped runtime "guard" that temporarily patches
``multiprocessing`` and ``concurrent.futures.process`` entry points while a
notebook test runs. The guard validates worker callables before dispatch when
``spawn``/``forkserver`` is used, so notebook-defined functions fail fast with
a clear error instead of failing later with pickling/import noise.
"""

from __future__ import annotations

import concurrent.futures.process as concurrent_futures_process
import multiprocessing as mp
import multiprocessing.pool as multiprocessing_pool
import multiprocessing.process as multiprocessing_process
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, cast

_SYNC_EXEC_MODULE_NAME = "pytest_nb_as_test.__nbast_main__"
_MAIN_MODULE_NAME = "__main__"
_SPAWN_START_METHODS = frozenset({"spawn", "forkserver"})
_SPAWN_GUARDRAIL_MESSAGE = (
    "spawn cannot pickle notebook defined callables, "
    "put worker targets in an importable .py module"
)


def _ensure_spawn_bootstrap_script() -> str:
    """Create a no-op script used as a safe spawn bootstrap target.

    Returns:
        Absolute path to a Python script that executes without side effects.

    Example:
        bootstrap_path = _ensure_spawn_bootstrap_script()
    """
    bootstrap_dir = Path(tempfile.gettempdir()) / "pytest-nb-as-test"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_path = bootstrap_dir / "spawn_bootstrap.py"
    if not bootstrap_path.exists():
        bootstrap_path.write_text(
            '"""No-op bootstrap script for multiprocessing spawn."""\n\n'
            'if __name__ == "__main__":\n'
            "    pass\n",
            encoding="utf-8",
        )
    return str(bootstrap_path)


def _resolve_safe_main_file_for_spawn() -> str:
    """Return a script path safe for spawn/forkserver bootstrap.

    Multiprocessing ``spawn`` and ``forkserver`` bootstrap child processes by
    executing ``__main__.__file__``. During notebook execution, ``__main__``
    points at the synthetic notebook module, so exposing the ``.ipynb`` path
    there causes children to run notebook JSON as Python.

    Returns:
        Path to a script that child processes can safely execute.

    Example:
        safe_file = _resolve_safe_main_file_for_spawn()
    """
    current_main = sys.modules.get(_MAIN_MODULE_NAME)
    for candidate in (getattr(current_main, "__file__", None), sys.argv[0]):
        if not isinstance(candidate, str) or not candidate:
            continue
        candidate_path = os.path.abspath(candidate)
        if os.path.isfile(candidate_path) and not candidate_path.lower().endswith(
            ".ipynb"
        ):
            return candidate_path
    return _ensure_spawn_bootstrap_script()


class _NotebookPoolProxy:  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Proxy around ``multiprocessing`` pools with callable validation.

    Example:
        proxy = _NotebookPoolProxy(pool, "spawn", validator)
    """

    def __init__(
        self,
        pool: Any,
        start_method: str,
        validator: Callable[[Callable[..., Any], str], None],
    ) -> None:
        self._pool = pool
        self._start_method = start_method
        self._validator = validator

    def _check_callable(self, func: Callable[..., Any]) -> None:
        """Validate a worker callable before dispatch.

        Args:
            func: Worker callable submitted to the pool.

        Example:
            self._check_callable(worker)
        """
        self._validator(func, self._start_method)

    def apply(
        self, func: Callable[..., Any], args: Any = (), kwds: Any | None = None
    ) -> Any:
        """Proxy ``Pool.apply`` with validation.

        Example:
            result = proxy.apply(worker, args=(1,))
        """
        self._check_callable(func)
        return self._pool.apply(func, args=args, kwds=kwds)

    def apply_async(
        self,
        func: Callable[..., Any],
        args: Any = (),
        kwds: Any | None = None,
        callback: Callable[..., Any] | None = None,
        error_callback: Callable[..., Any] | None = None,
    ) -> Any:
        """Proxy ``Pool.apply_async`` with validation.

        Example:
            async_result = proxy.apply_async(worker, args=(1,))
        """
        self._check_callable(func)
        return self._pool.apply_async(
            func,
            args=args,
            kwds=kwds,
            callback=callback,
            error_callback=error_callback,
        )

    def map(
        self,
        func: Callable[..., Any],
        iterable: Any,
        chunksize: int | None = None,
    ) -> Any:
        """Proxy ``Pool.map`` with validation.

        Example:
            out = proxy.map(worker, [1, 2, 3])
        """
        self._check_callable(func)
        return self._pool.map(func, iterable, chunksize=chunksize)

    def map_async(
        self,
        func: Callable[..., Any],
        iterable: Any,
        chunksize: int | None = None,
        callback: Callable[..., Any] | None = None,
        error_callback: Callable[..., Any] | None = None,
    ) -> Any:
        """Proxy ``Pool.map_async`` with validation.

        Example:
            out = proxy.map_async(worker, [1, 2, 3])
        """
        self._check_callable(func)
        return self._pool.map_async(
            func,
            iterable,
            chunksize=chunksize,
            callback=callback,
            error_callback=error_callback,
        )

    def starmap(
        self,
        func: Callable[..., Any],
        iterable: Any,
        chunksize: int | None = None,
    ) -> Any:
        """Proxy ``Pool.starmap`` with validation.

        Example:
            out = proxy.starmap(worker, [(1,), (2,)])
        """
        self._check_callable(func)
        return self._pool.starmap(func, iterable, chunksize=chunksize)

    def starmap_async(
        self,
        func: Callable[..., Any],
        iterable: Any,
        chunksize: int | None = None,
        callback: Callable[..., Any] | None = None,
        error_callback: Callable[..., Any] | None = None,
    ) -> Any:
        """Proxy ``Pool.starmap_async`` with validation.

        Example:
            out = proxy.starmap_async(worker, [(1,), (2,)])
        """
        self._check_callable(func)
        return self._pool.starmap_async(
            func,
            iterable,
            chunksize=chunksize,
            callback=callback,
            error_callback=error_callback,
        )

    def imap(
        self,
        func: Callable[..., Any],
        iterable: Any,
        chunksize: int = 1,
    ) -> Any:
        """Proxy ``Pool.imap`` with validation.

        Example:
            out = proxy.imap(worker, [1, 2, 3])
        """
        self._check_callable(func)
        return self._pool.imap(func, iterable, chunksize=chunksize)

    def imap_unordered(
        self,
        func: Callable[..., Any],
        iterable: Any,
        chunksize: int = 1,
    ) -> Any:
        """Proxy ``Pool.imap_unordered`` with validation.

        Example:
            out = proxy.imap_unordered(worker, [1, 2, 3])
        """
        self._check_callable(func)
        return self._pool.imap_unordered(func, iterable, chunksize=chunksize)

    def __enter__(self) -> _NotebookPoolProxy:
        """Enter the proxied pool context.

        Example:
            with proxy as active:
                ...
        """
        self._pool.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> Any:
        """Exit the proxied pool context.

        Example:
            proxy.__exit__(None, None, None)
        """
        return self._pool.__exit__(exc_type, exc, tb)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the underlying pool.

        Args:
            name: Attribute name.

        Example:
            state = proxy._state
        """
        return getattr(self._pool, name)


class _NotebookContextProxy:
    """Proxy around ``multiprocessing`` context objects.

    Example:
        guarded = _NotebookContextProxy(ctx, "spawn", validator)
    """

    def __init__(
        self,
        context: Any,
        start_method: str,
        validator: Callable[[Callable[..., Any], str], None],
    ) -> None:
        self._context = context
        self._start_method = start_method
        self._validator = validator

    def Pool(  # pylint: disable=invalid-name
        self, *args: Any, **kwargs: Any
    ) -> _NotebookPoolProxy:
        """Create a guarded pool from the wrapped context.

        Args:
            *args: Positional arguments for ``context.Pool``.
            **kwargs: Keyword arguments for ``context.Pool``.

        Returns:
            Guarded pool proxy.

        Example:
            pool = guarded.Pool(processes=2)
        """
        pool = self._context.Pool(*args, **kwargs)
        return _NotebookPoolProxy(pool, self._start_method, self._validator)

    def get_start_method(self, allow_none: bool = False) -> str:
        """Return the wrapped context start method.

        Args:
            allow_none: Compatibility argument used by stdlib context APIs.

        Example:
            mode = guarded.get_start_method()
        """
        del allow_none
        return self._start_method

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped context.

        Args:
            name: Attribute name.

        Example:
            proc = guarded.Process
        """
        return getattr(self._context, name)


@contextmanager
def _spawn_guard_context(  # pylint: disable=too-many-locals
    validator: Callable[[Callable[..., Any], str], None],
    used_start_methods: set[str],
) -> Iterator[None]:
    """Patch multiprocessing entry points to validate spawn worker callables.

    The guard intercepts process and pool submission paths and calls
    ``validator`` with the submitted callable and active start method. This is
    used to block notebook-defined callables under ``spawn``/``forkserver``,
    where child interpreters require importable module-level targets.

    Args:
        validator: Callback used to validate submitted worker callables.
        used_start_methods: Collector for observed start methods.

    Yields:
        None.

    Example:
        with _spawn_guard_context(validator, methods):
            ...
    """
    original_get_context = mp.get_context
    original_pool = mp.Pool
    pool_method_names = (
        "apply",
        "apply_async",
        "map",
        "map_async",
        "starmap",
        "starmap_async",
        "imap",
        "imap_unordered",
    )
    original_pool_methods = {
        name: getattr(multiprocessing_pool.Pool, name) for name in pool_method_names
    }
    original_process_start = multiprocessing_process.BaseProcess.start
    original_executor_submit = concurrent_futures_process.ProcessPoolExecutor.submit
    original_executor_map = concurrent_futures_process.ProcessPoolExecutor.map
    process_pool_executor_cls = cast(
        Any,
        concurrent_futures_process.ProcessPoolExecutor,
    )

    def _resolve_start_method(context: Any | None = None) -> str:
        if context is None:
            return cast(str, original_get_context().get_start_method())
        return cast(str, context.get_start_method())

    def _record_start_method(start_method: str) -> str:
        used_start_methods.add(start_method)
        return start_method

    def _resolve_pool_start_method(pool: Any) -> str:
        pool_context = getattr(pool, "_ctx", None)
        if pool_context is None:
            return _record_start_method(_resolve_start_method())
        return _record_start_method(_resolve_start_method(pool_context))

    def _resolve_process_start_method(process: Any) -> str:
        process_start_method = getattr(type(process), "_start_method", None)
        if isinstance(process_start_method, str):
            return _record_start_method(process_start_method)
        process_context = getattr(process, "_ctx", None)
        if process_context is not None:
            return _record_start_method(_resolve_start_method(process_context))
        return _record_start_method(_resolve_start_method())

    def guarded_get_context(method: str | None = None) -> Any:
        context = original_get_context(method)
        _record_start_method(cast(str, context.get_start_method()))
        return context

    def guarded_pool(*args: Any, **kwargs: Any) -> Any:
        _record_start_method(_resolve_start_method())
        return original_pool(*args, **kwargs)

    def guarded_process_start(self: Any) -> Any:
        target = getattr(self, "_target", None)
        start_method = _resolve_process_start_method(self)
        if callable(target):
            validator(cast(Callable[..., Any], target), start_method)
        return original_process_start(self)

    def _make_guarded_pool_method(name: str) -> Callable[..., Any]:
        original_method = original_pool_methods[name]

        def guarded_method(
            self: multiprocessing_pool.Pool,
            func: Callable[..., Any],
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            validator(func, _resolve_pool_start_method(self))
            return original_method(self, func, *args, **kwargs)

        return guarded_method

    def guarded_executor_submit(
        self: concurrent_futures_process.ProcessPoolExecutor,
        fn: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        context = getattr(self, "_mp_context", None)
        validator(fn, _record_start_method(_resolve_start_method(context)))
        return original_executor_submit(self, fn, *args, **kwargs)

    def guarded_executor_map(
        self: concurrent_futures_process.ProcessPoolExecutor,
        fn: Callable[..., Any],
        /,
        *iterables: Any,
        **kwargs: Any,
    ) -> Any:
        context = getattr(self, "_mp_context", None)
        validator(fn, _record_start_method(_resolve_start_method(context)))
        return original_executor_map(self, fn, *iterables, **kwargs)

    try:
        mp.get_context = guarded_get_context
        mp.Pool = guarded_pool
        multiprocessing_process.BaseProcess.start = guarded_process_start
        for method_name in pool_method_names:
            setattr(
                multiprocessing_pool.Pool,
                method_name,
                _make_guarded_pool_method(method_name),
            )
        process_pool_executor_cls.submit = guarded_executor_submit
        process_pool_executor_cls.map = guarded_executor_map
        yield
    finally:
        mp.get_context = original_get_context
        mp.Pool = original_pool
        multiprocessing_process.BaseProcess.start = original_process_start
        for method_name, original_method in original_pool_methods.items():
            setattr(multiprocessing_pool.Pool, method_name, original_method)
        process_pool_executor_cls.submit = original_executor_submit
        process_pool_executor_cls.map = original_executor_map
