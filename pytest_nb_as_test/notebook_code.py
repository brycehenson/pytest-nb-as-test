"""Helpers for selecting notebook cells and generating test code."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class SelectedCell:  # pylint: disable=too-few-public-methods
    """Selected code cell with execution metadata.

    Example:
        cell = SelectedCell(
            index=1,
            source="print('hello')",
            must_raise=False,
            timeout_seconds=None,
        )
    """

    index: int
    """Cell index within the notebook."""

    source: str
    """Cell source code."""

    must_raise: bool
    """Whether the cell is expected to raise an exception."""

    timeout_seconds: float | None
    """Per-cell timeout override in seconds."""


@dataclass(frozen=True, kw_only=True)
class CellCodeSpan:  # pylint: disable=too-few-public-methods
    """Mapping between generated code lines and a notebook cell.

    Example:
        span = CellCodeSpan(
            index=2,
            block_start_line=14,
            block_end_line=21,
            cell_start_line=16,
            cell_end_line=18,
            source="print('hello')",
        )
    """

    index: int
    """Cell index within the notebook."""

    block_start_line: int
    """First generated line for the cell block, including wrapper context."""

    block_end_line: int
    """Last generated line for the cell block, including wrapper context."""

    cell_start_line: int
    """First generated line corresponding to the cell's code."""

    cell_end_line: int
    """Last generated line corresponding to the cell's code."""

    source: str
    """Transformed cell source code that was executed."""


def _comment_out_ipython_magics(source: str) -> str:
    """Comment out IPython magics and shell escapes in a code cell.

    Args:
        source: Cell source code to be transformed.

    Returns:
        Transformed source with IPython line magics (``%``), cell magics (``%%``),
        and shell escapes (``!``) commented out.

    Example:
        _comment_out_ipython_magics("%time\\nx = 1\\n")
    """
    if re.search(r"(^ {0,5})%%", source, flags=re.MULTILINE):
        commented_lines: list[str] = []
        for line in source.splitlines(keepends=True):
            if not line.strip():
                commented_lines.append(line)
                continue
            commented_lines.append(re.sub(r"^([ \\t]*)", r"\1#", line, count=1))
        return "".join(commented_lines)

    return re.sub(
        r"(^ {0,5})([%!])",
        lambda m: m.group(1) + "#" + m.group(2),
        source,
        flags=re.MULTILINE,
    )


def _extract_future_imports(source: str) -> tuple[list[str], str]:
    """Extract future import statements from notebook cell source.

    Args:
        source: Raw cell source code.

    Returns:
        A tuple of future import lines (no trailing newlines) and the remaining
        cell source with those lines removed.

    Example:
        future_lines, rest = _extract_future_imports(
            "from __future__ import annotations\\nprint('hi')\\n"
        )
    """
    future_lines: list[str] = []
    remaining_lines: list[str] = []
    for line in source.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("from __future__ import "):
            future_lines.append(stripped.rstrip("\r\n"))
        else:
            remaining_lines.append(line)
    return future_lines, "".join(remaining_lines)
