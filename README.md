# pytest-notebook-test Plugin
[![CI pipeline status](https://github.com/brycehenson/pytest_notebook/actions/workflows/ci.yml/badge.svg)](https://github.com/brycehenson/pytest_notebook/actions/workflows/ci.yml)


In scientific codebases, notebooks are a convenient way to provide executable examples, figures, and LaTeX.
However, example notebooks often become silently broken as the code evolves because developers rarely re run them.
New users then discover the breakage when they try the examples, which is disheartening and frustrating.
This plugin executes notebook code cells as `pytest` tests, so example notebooks run in CI and stay up to date.

## When to use
- You want `.ipynb` notebooks collected by pytest and run in CI.
- You want in process execution, so fixtures and monkeypatching apply.
- You need per cell control (skip, force run, expect exception, timeouts) via directives.


## Install
Add a dependency (example `pyproject.toml`):

```toml
[project]
dependencies = [
  "pytest-notebook-test @ git+https://github.com/brycehenson/pytest_notebook@main",
]
```

## Run

Pytest discovers all notebooks alongside normal tests:

```bash
pytest
```

Filter which notebooks are collected:

```bash
pytest --notebook-glob 'test_*.ipynb'
```

Disable notebook collection and execution:

```bash
pytest -p no:pytest_notebook_test
```

## Cell directives

Directives live in comments inside *code* cells.
They are ignored in markdown cells.

General form:

```python
# notebook-test: <flag>=<value>
```

Rules:

* each flag may appear at most once per cell
* booleans accept `True` or `False` (case sensitive)
* timeouts accept numeric seconds
* invalid values, or repeated flags, fail at collection time


### `default-all`

Sets the default inclusion status for subsequent code cells.

```python
# notebook-test: default-all=True|False
```

Example:

```python
# notebook-test: default-all=False
# cells from here are skipped

# ... plotting, exploration, notes ...

# notebook-test: default-all=True
# execution resumes
```

### `test-cell`

Overrides the current default for the current cell only.

```python
# notebook-test: test-cell=True|False
```

### `must-raise-exception`

Marks a cell as expected to raise an exception.

```python
# notebook-test: must-raise-exception=True|False
```

If `True`, the cell is executed under `pytest.raises(Exception)`.
The test fails if no exception is raised, or if a `BaseException` (for example `SystemExit`) is raised.

Example:

```python
# notebook-test: must-raise-exception=True
raise ValueError("Intentional failure for demonstration")
```

### `notebook-timeout-seconds`

Sets a wall clock timeout (seconds) for the whole notebook.
Requires `pytest-timeout`.
Must appear in the first code cell.

```python
# notebook-test: notebook-timeout-seconds=<float>
```

### `cell-timeout-seconds`

Sets a per cell timeout (seconds).
Requires `pytest-timeout`.

```python
# notebook-test: cell-timeout-seconds=<float>
```

## Configuration

Precedence order:

1. In notebook directives
2. CLI options
3. `pytest.ini` or `pyproject.toml`
4. environment variables (if supported by your config)
5. defaults

### CLI options


| Option | Type | Default | Description |
|---|---|---:|---|
| `--notebook-default-all` | `true` `false` | `true` | Initial value of the `test_all_cells` flag. If `false` then cells without an explicit `test-cell` directive will be skipped until `default-all=True` is encountered. |
| `--notebook-glob <pattern>` | string | `none` | Glob pattern for notebook filenames, name-only patterns match basenames, path patterns match relative paths. |
| `--notebook-keep-generated` | `none` `onfail` `<path>`  | `onfail` | Controls dumping of the generated test script. `none` means never dump, `onfail` dumps the script into the report upon a test failure, any other string is treated as a path and the script is written there with a filename derived from the notebook name. |
| `--notebook-exec-mode {async,sync}` | `async` `sync` | `async` | Whether to generate `async def` or `def` for the wrapper. If `async`, the plugin marks the test item with `pytest.mark.asyncio` if the `pytest-asyncio` plugin is installed. If `sync`, the code runs synchronously. |
| `--notebook-timeout-seconds` | float | `none` | Wall-clock timeout for an entire notebook, enforced via `pytest-timeout`. |
| `--notebook-cell-timeout-seconds` | float | `none` | Default per-cell timeout in seconds, enforced via `pytest-timeout`. |


### pytest.ini / pyproject.toml settings

You can set options in your `pytest.ini` or `pyproject.toml` under
`[tool.pytest.ini_options]`.  For example:

```ini
[pytest]
notebook_default_all = false
notebook-timeout-seconds = 120
notebook-cell-timeout-seconds= 10

```

Values set in the ini file are overridden by command line arguments and
environment variables as described above.

In `pyproject.toml`, put the same keys under `[tool.pytest.ini_options]`.


## Debugging failures

On failure, the plugin can attach the generated Python script to the pytest report.
With `--notebook-keep-generated=onfail` (default) you get a “generated notebook script” section in the report.

If you pass a directory to `--notebook-keep-generated`, the script is written there with a name derived from the notebook filename.

Each selected cell is preceded by a marker comment:

```python
## notebook-test notebook=<filename> cell=<index>
```

Use this to correlate tracebacks with notebook cell indices.

## Demo

Run the demo harness:

```bash
python run_demo.py
```

It copies a small set of notebooks into a temporary workspace, invokes pytest, and reports outcomes.

## Development and testing

The plugin tests live in `tests/test_plugin.py` and use notebooks under `tests/notebooks/`.

Run:

```bash
pytest
```

Examples:

```bash
pytest tests/notebooks/example_simple_123.ipynb
pytest tests/notebooks --notebook-glob "test_*.ipynb"
```



## Suggested conftest snippets

Put these in a `conftest.py` near your notebooks and keep them scoped to
notebook tests via the `notebook` marker.

### NumPy RNG: seed and ensure it is unused

```python
import pytest


@pytest.fixture(autouse=True)
def seed_and_lock_numpy_rng(request: pytest.FixtureRequest) -> None:
    if request.node.get_closest_marker("notebook") is None:
        yield
        return

    try:
        import numpy as np
    except ModuleNotFoundError:
        yield
        return

    np.random.seed(0)
    state = np.random.get_state()
    yield
    new_state = np.random.get_state()

    same_state = (
        state[0] == new_state[0]
        and state[2:] == new_state[2:]
        and np.array_equal(state[1], new_state[1])
    )
    if not same_state:
        raise AssertionError("NumPy RNG state changed; random was called.")
```

### Matplotlib backend

```python
import pytest


@pytest.fixture(autouse=True)
def set_matplotlib_backend(request: pytest.FixtureRequest) -> None:
    if request.node.get_closest_marker("notebook") is None:
        yield
        return

    try:
        import matplotlib
    except ModuleNotFoundError:
        yield
        return

    matplotlib.use("Agg")
    yield
```

### Plotly renderer

```python
import pytest


@pytest.fixture(autouse=True)
def set_plotly_renderer(request: pytest.FixtureRequest) -> None:
    if request.node.get_closest_marker("notebook") is None:
        yield
        return

    try:
        import plotly.io as pio
    except ModuleNotFoundError:
        yield
        return

    pio.renderers.default = "jpg"
    yield
```
