# Codex Working Notes

These notes capture repository-wide expectations for future automation runs once the template is customised.

## Execution Modes
- Treat `sandbox_mode=read-only` as "chat mode".
- In chat mode, do not edit files. Instead, provide:
  - A clear plan of the changes you would make.
  - A detailed implementation report covering file paths, specific code locations, exact edits, new/changed functions or classes, data structures, and any assumptions.
  - The validation steps you would run (lint/tests/notebooks), plus any expected outputs or skips.
- If edits would require approvals, call that out explicitly after the plan.

## Coding Standards
- Prefer full type hints and concise module-level docstrings.
- Keep public and private functions/classes documented with Google-style docstrings, include an "Example" heading which shows the basic usage of a function or method.
- Document private methods that implement non-trivial logic or define subclass contracts, using the same Args/Returns/Example style as public methods.
- Isolate side effects at the service layer; keep utilities and pipelines pure so they are easy to test.
- Avoid using private APIs from other packages; rely on documented, public interfaces unless explicitly required and approved.
- Avoid very short internal helper functions with trivial logic used only once; keep that code inline at the call site and add a brief comment when it improves clarity.
- Avoid raw string options; use `Enum` or `Literal` for finite option sets and convert/validate early.
- Use enums or dataclasses when a group of related values travels together instead of loosely-coupled dictionaries.
- Prefer dataclasses (often `frozen=True`) for structured results over dict outputs; only use dicts when required by external APIs and document expected keys.
- Provide a docstring for every dataclass field, placed immediately below the field definition. e.g:
```
@dataclass(frozen=True, kw_only=True)
class LocalFieldSource(FieldSource, ABC):
    """Field source with a local coordinate system and world placement."""

    center: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    """World-space center of the source in metres."""
```
- Include array shapes and physical units in docstrings; use unit suffixes in variable names when practical (e.g., `_T`, `_m2`, `_Hz`).
- Validate shapes of numpy arrays explicitly, and raise `ValueError` on mismatch.
- When iterating, pre-allocate numpy arrays with the correct shape and fill them with `np.nan`.
- Use `__post_init__` for validation/normalization in dataclasses; use `object.__setattr__` in frozen dataclasses.
- Use `numpy.typing` (`npt.NDArray[...]`) for array-heavy APIs and keep full type hints.

## Repository Conventions
- Maintain the `src/` + `tests/` structure even if you rename the package.
- Add linting/formatting tools under `[project.optional-dependencies]` (e.g. `dev`) so they are opt-in per consumer.

## Validation Workflow
- For any Python code edits, run `pylint` first on the touched modules (e.g. `pylint src/mag_trap/bfield_sources.py`), fix issues before proceeding, and report the results.
- If you create or edit runnable scripts or modules, run a minimal invocation of the code to validate it executes.
- If you create or edit notebooks, execute them end-to-end and report the outcome.
- Always report lint/test/notebook execution results in the final response, and explicitly call out any skips with reasons.
