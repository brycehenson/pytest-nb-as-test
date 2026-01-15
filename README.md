# pytest-nb-as-test Plugin
[![CI pipeline status](https://github.com/brycehenson/pytest-nb-as-test/actions/workflows/ci.yml/badge.svg?job=pytest)](https://github.com/brycehenson/pytest-nb-as-test/actions/workflows/ci.yml)

![icon](https://github.com/brycehenson/pytest-nb-as-test/blob/main/icon.png)


In scientific codebases, notebooks are a convenient way to provide executable examples, figures, and LaTeX.
However, example notebooks often become silently broken as the code evolves because developers rarely re-run them.
New users then discover the breakage when they try the examples, which is disheartening and frustrating.
This plugin executes notebook code cells as `pytest` tests, so example notebooks run in CI and stay up to date.

## When to use
- You want `.ipynb` notebooks collected by pytest and run in CI.
- You want in process execution, so fixtures and monkeypatching apply.
- You need per cell control (skip, force run, expect exception, timeouts) via directives.
  
For comparison with other plugins/ projects see [Prior art and related tools](#prior-art-and-related-tools).


## Install
install using pip

```bash
pip install pytest-nb-as-test
```
or add as dependency in `pyproject.toml`:
```toml
[project]
dependencies = [
  "pytest-nb-as-test",
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
pytest -p no:pytest_nb_as_test
```

## Cell directives

Directives live in comments inside *code* cells.
They are ignored in markdown cells.

General form:

```python
# pytest-nb-as-test: <flag>=<value>
```

Rules:

* each flag may appear at most once per cell
* booleans accept `True` or `False` (case sensitive)
* timeouts accept numeric seconds
* invalid values, or repeated flags, fail at collection time


### `default-all`

Sets the default inclusion status for subsequent code cells.

```python
# pytest-nb-as-test: default-all=True|False
```

Example:

```python
# pytest-nb-as-test: default-all=False
# cells from here are skipped

# ... plotting, exploration, notes ...

# pytest-nb-as-test: default-all=True
# execution resumes
```

### `test-cell`

Overrides the current default for the current cell only.

```python
# pytest-nb-as-test: test-cell=True|False
```

### `must-raise-exception`

Marks a cell as expected to raise an exception.

```python
# pytest-nb-as-test: must-raise-exception=True|False
```

If `True`, the cell is executed under `pytest.raises(Exception)`.
The test fails if no exception is raised, or if a `BaseException` (for example `SystemExit`) is raised.

Example:

```python
# pytest-nb-as-test: must-raise-exception=True
raise ValueError("Intentional failure for demonstration")
```

### `notebook-timeout-seconds`

Sets a wall clock timeout (seconds) for the whole notebook.
Requires `pytest-timeout`.
Must appear in the first code cell.

```python
# pytest-nb-as-test: notebook-timeout-seconds=<float>
```

### `cell-timeout-seconds`

Sets a per cell timeout (seconds).
Requires `pytest-timeout`.

```python
# pytest-nb-as-test: cell-timeout-seconds=<float>
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
| `--notebook-glob` | string | `none` | Glob pattern for notebook filenames, name-only patterns match basenames, path patterns match relative paths. |
| `--notebook-keep-generated` | `none` `onfail` `<path>`  | `onfail` | Controls dumping of the generated test script. `none` means never dump, `onfail` dumps the script into the report upon a test failure, any other string is treated as a path and the script is written there with a filename derived from the notebook name. |
| `--notebook-exec-mode` | `async` `sync` | `async` | Whether to generate `async def` or `def` for the wrapper. If `async`, the plugin marks the test item with `pytest.mark.asyncio` and uses the pytest-asyncio event loop when the plugin is installed; otherwise it runs the coroutine with `asyncio.run()`. If `sync`, the code runs synchronously. |
| `--notebook-timeout-seconds` | float | `none` | Wall-clock timeout for an entire notebook, enforced via `pytest-timeout`. |
| `--notebook-cell-timeout-seconds` | float | `none` | Default per-cell timeout in seconds, enforced via `pytest-timeout`. |


### pytest.ini / pyproject.toml settings

You can set options in your `pytest.ini` or `pyproject.toml` under
`[tool.pytest.ini_options]`. In ini files, use the underscore option names
(`notebook_default_all`), not the CLI flag form with dashes. For example:

```ini
[pytest]
notebook_default_all = false
notebook_timeout_seconds = 120
notebook_cell_timeout_seconds = 10

```

Values set in the ini file are overridden by command line arguments and
environment variables as described above.

In `pyproject.toml`, put the same keys under `[tool.pytest.ini_options]`.

Note: `notebook_default_all = false` only changes which cells are selected
inside notebooks; it does not disable notebook collection. To skip notebook
tests entirely, use `-m "not notebook"` or `--ignore-glob=*.ipynb` in `addopts`.


## Debugging failures

On failure, the plugin can attach the generated Python script to the pytest report.
With `--notebook-keep-generated=onfail` (default) you get a “generated notebook script” section in the report.

If you pass a directory to `--notebook-keep-generated`, the script is written there with a name derived from the notebook filename.

Each selected cell is preceded by a marker comment:

```python
## pytest-nb-as-test notebook=<filename> cell=<index>
```

Use this to correlate tracebacks with notebook cell indices.

## Versioning / API stability

This project follows Semantic Versioning.

Before 1.0, public APIs may change without notice.
After 1.0, the following are considered stable public APIs:

- CLI options listed in this README.
- `pytest.ini` / `pyproject.toml` configuration keys listed in this README.
- Notebook directives (`default-all`, `test-cell`, `must-raise-exception`, `notebook-timeout-seconds`, `cell-timeout-seconds`).

Behavioral changes to these APIs will be announced in the changelog and, when practical,
introduced with a deprecation period of at least one minor release.

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

    os.environ.setdefault("PLOTLY_RENDERER", "json")

    import plotly.io as pio

    pio.renderers.default = "json"
    pio.renderers.render_on_display = False
    pio.show = lambda *args, **kwargs: None
    yield

```



## Prior art and related tools

Several existing projects test notebooks, but they optimise for different goals.

### Output regression testing (compare stored outputs)
- **nbval**: collects notebooks, executes them in a Jupyter kernel, and compares executed cell outputs against the outputs stored in the `.ipynb` (each cell behaves like a test). It also supports output sanitisation for noisy outputs.  
  https://pypi.org/project/nbval/ 
- **pytest-notebook**: executes notebooks, diffs input vs output notebooks (via `nbdime`), and can regenerate notebooks when outputs change. Also integrates with coverage tooling.  
  https://pytest-notebook.readthedocs.io/ 
**When to prefer these:** you want to detect changes in rendered outputs, not just “runs without error”.

### Execute notebooks under pytest (smoke execution, not output diffs)
- **pytest-nbmake**: executes notebooks during pytest using `nbclient`. Supports per-cell behaviour via notebook cell tags (for example `skip-execution`, `raises-exception`).  
  https://github.com/treebeardtech/pytest-nbmake 
**When to prefer this:** you want faithful notebook execution semantics (kernel based execution) and simple CI integration.

### “Tests inside notebooks” (interactive and teaching workflows)
- **pytest-ipynb2**: collects tests written in notebooks via a `%%ipytest` magic, supports fixtures and parametrisation, and executes cells above the test cell.  
  https://musicalninjadad.github.io/pytest-ipynb2/ 
- **ipytest**: run pytest conveniently from within a notebook (primarily interactive UX).  
  https://github.com/chmp/ipytest 
- **nbtest-plugin**: provides notebook-friendly assertion helpers (including DataFrame assertions) that are later collected by pytest when run with `--nbtest`.  
  https://pypi.org/project/nbtest-plugin/ 
- **nbcelltests**: cell-by-cell testing aimed at “linearly executed notebooks”, with JupyterLab integration.  
  https://github.com/jpmorganchase/nbcelltests 
  It integrates with **JupyterLab** via bundled lab and server extensions, so tests can be authored and run from the browser.
  Tests are stored in **cell metadata**, and nbcelltests generates a Python `unittest` class with per cell methods whose state includes the cumulative context of all prior cells, mimicking linear execution.
  Inside a test you can use `%cell` to inject the corresponding notebook cell source into the generated test method.
  It can also run offline from an `.ipynb`, and it supports a lint mode plus additional structural checks such as maximum lines per cell, maximum cells per notebook, maximum number of function or class definitions, and minimum percentage of cells tested. 

### How `pytest-nb-as-test` differs

This plugin is aimed at *CI enforcement of example notebooks* in scientific codebases, with two deliberate design choices:

1. **In-process execution** so that normal pytest mechanisms (fixtures, `monkeypatch`, markers) can apply to notebook code.
2. **Per-cell directives embedded in code cell comments** (`default-all`, `test-cell`, timeouts, expected exceptions), so behaviour is visible in diffs without relying on notebook metadata.

If you need output regression diffs, prefer `nbval` or `pytest-notebook`.
If you need faithful kernel execution semantics, prefer `pytest-nbmake`.
