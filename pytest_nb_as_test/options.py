"""Option parsing and configuration helpers for the pytest plugin."""

from __future__ import annotations

import os
from typing import Any

import pytest  # type: ignore


def _parse_bool(value: str) -> bool:
    """Parse a boolean option from a string.

    Args:
        value: Raw string value.

    Returns:
        Parsed boolean value.

    Example:
        is_enabled = _parse_bool("true")
    """
    val = value.strip().lower()
    if val in {"true", "1", "yes", "on"}:
        return True
    if val in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _parse_timeout_seconds(value: str, where: str) -> float:
    """Parse a timeout value in seconds.

    Args:
        value: Raw string value to parse.
        where: Location description for error messages.

    Returns:
        Timeout value in seconds.

    Example:
        seconds = _parse_timeout_seconds("0.5", "cell directive")
    """
    try:
        seconds = float(value)
    except ValueError as exc:
        raise pytest.UsageError(
            f"Invalid timeout value {value!r} from {where}."
        ) from exc
    if seconds <= 0:
        raise pytest.UsageError(
            f"Timeout value {value!r} from {where} must be > 0 seconds."
        )
    return seconds


def _parse_optional_timeout(value: Any, where: str) -> float | None:
    """Parse an optional timeout value in seconds.

    Args:
        value: Raw value or None.
        where: Location description for error messages.

    Returns:
        Timeout value in seconds, or None if not provided.

    Example:
        timeout = _parse_optional_timeout(None, "ini config")
    """
    if value in (None, "", []):
        return None
    return _parse_timeout_seconds(str(value), where)


def _cli_flag_present(config: pytest.Config, flag: str) -> bool:
    """Return True if a CLI flag is present in the invocation args.

    Args:
        config: Pytest configuration object.
        flag: Long-form CLI flag, e.g. ``--notebook-keep-generated``.

    Returns:
        True when the flag is present as ``--flag`` or ``--flag=value``.

    Example:
        if _cli_flag_present(config, "--notebook-keep-generated"):
            ...
    """
    for arg in config.invocation_params.args:
        if arg == flag or arg.startswith(f"{flag}="):
            return True
    return False


def _resolve_option(
    config: pytest.Config,
    name: str,
    env_var: str | None = None,
    default: Any | None = None,
    cli_flag: str | None = None,
) -> Any:
    """Resolve an option by checking CLI args, ini settings, and environment.

    Args:
        config: Pytest configuration object.
        name: Base name of the option, e.g. ``notebook_default_all``.
        env_var: Optional environment variable name.
        default: Default value when no configuration is supplied.
        cli_flag: Optional CLI flag name to verify presence.

    Returns:
        The resolved option value.

    Example:
        keep_generated = _resolve_option(
            config,
            "notebook_keep_generated",
            default="onfail",
            cli_flag="--notebook-keep-generated",
        )
    """
    # command line overrides ini and env
    try:
        cli_value = config.getoption(name)
    except ValueError:
        cli_value = None
    if cli_flag is not None and not _cli_flag_present(config, cli_flag):
        cli_value = None
    if cli_value is not None:
        return cli_value
    # ini next
    try:
        ini_value = config.getini(name)
    except ValueError:
        ini_value = None
    if ini_value not in (None, "", []):
        return ini_value
    # environment variable if provided
    if env_var is not None:
        env_value = os.getenv(env_var)
        if env_value:
            return env_value
    return default


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register command line options and ini variables for the plugin.

    Args:
        parser: Pytest command line parser.

    Example:
        pytest_addoption(parser)
    """
    group = parser.getgroup("notebook-test")
    group.addoption(
        "--notebook-default-all",
        action="store",
        dest="notebook_default_all",
        default=None,
        help="Initial default for test_all_cells (true/false).",
    )
    group.addoption(
        "--notebook-glob",
        action="store",
        dest="notebook_glob",
        default=None,
        help=(
            "Glob pattern for notebook files; applies to --notebook-dir or all discovered "
            "notebooks."
        ),
    )
    group.addoption(
        "--notebook-keep-generated",
        action="store",
        dest="notebook_keep_generated",
        default="onfail",
        help="Control dumping of generated test script: 'none', 'onfail' or directory path.",
    )
    group.addoption(
        "--notebook-exec-mode",
        action="store",
        dest="notebook_exec_mode",
        default=None,
        help="Execution mode for notebooks: 'async' (default) or 'sync'.",
    )
    group.addoption(
        "--notebook-timeout-seconds",
        action="store",
        dest="notebook_timeout_seconds",
        default=None,
        help="Timeout for an entire notebook in seconds (requires pytest-timeout).",
    )
    group.addoption(
        "--notebook-cell-timeout-seconds",
        action="store",
        dest="notebook_cell_timeout_seconds",
        default=None,
        help="Default per-cell timeout in seconds (requires pytest-timeout).",
    )

    # Register ini options to allow configuration via pytest.ini or pyproject.toml
    parser.addini(
        "notebook_default_all",
        default="true",
        help="Initial default for test_all_cells (true/false).",
    )
    parser.addini(
        "notebook_glob",
        default="",
        help="Glob pattern for notebook files.",
    )
    parser.addini(
        "notebook_keep_generated",
        default="onfail",
        help="Dump generated code on failure or to a directory.",
    )
    parser.addini(
        "notebook_exec_mode",
        default="async",
        help="Execution mode for notebooks (async/sync).",
    )
    parser.addini(
        "notebook_timeout_seconds",
        default="",
        help="Timeout for an entire notebook in seconds (requires pytest-timeout).",
    )
    parser.addini(
        "notebook_cell_timeout_seconds",
        default="",
        help="Default per-cell timeout in seconds (requires pytest-timeout).",
    )
