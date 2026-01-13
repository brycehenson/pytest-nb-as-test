# pytest-notebook-test Plugin
[![CI pipeline status](https://github.com/brycehenson/pytest_notebook/actions/workflows/ci.yml/badge.svg)](https://github.com/brycehenson/pytest_notebook/actions/workflows/ci.yml)

## Use case

Use this plugin when you want Jupyter notebooks (`.ipynb`) to behave like
normal pytest tests.  It runs notebook cells in-process (no Jupyter kernel),
so your fixtures, monkeypatching, and test settings apply directly, while a
simple directive language lets you skip or enforce specific cells.

## Add as a package

Add the dependency to your project (example for `pyproject.toml`):

```toml
[project]
dependencies = [
  "pytest-notebook-test @ git+https://github.com/brycehenson/pytest_notebook@main",
]
```

## Run it

Run pytest as usual; notebooks are collected alongside your tests:

```bash
pytest
```
by default all notebooks are run (`*.ipynb`) to filter use `notebook-glob`
```bash
pytest  --notebook-glob 'test_*.ipynb'
```


disable this plugin, and just run non-notebook tests.
```bash
pytest -p no:pytest_notebook_test
```


## Notebook syntax
directives are controlled with in comments in notebook code cells 
```python
# notebook-test: <flag>=<value>
```

By default all notebook cells are tested, you can overide this in command line, pytest.ini or pyproject. You can also
set it in the notebook to turn on/off testing for the remainder of the notebook. Or use two commands to turn on/off
testing for a number of cells.
```python
# notebook-test: default-all=True|False
```

Overide the default for a single cell.
```python
# notebook-test: test-cell=True|False
```

Require a cell to raise any type of exception.
```python
# notebook-test: must-raise-exception=True
raise ValueError("Intentional failure for demonstration")
```

Set a per-cell timeout in seconds (requires pytest-timeout).
```python
# notebook-test: cell-timeout-seconds=0.5
```

Set a timeout for the entire notebook in seconds (requires pytest-timeout). This directive must be in the first code
cell in the notebook.
```python
# notebook-test: notebook-timeout-seconds=60
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

## Overview

This plugin turns Jupyter notebooks (`.ipynb` files) into first‑class pytest
test items.  It is based on the harness described in the task
but packaged as a proper pytest extension with a set of
configuration options.  It allows notebooks to be executed as tests within
the same Python process as your normal test suite while providing a
directive language for fine grained control over which cells are executed.

The key design goals of the plugin are:

* **In‑process execution** – notebook code is executed by compiling the code
  cells into a single function and running that function directly.  It does
  not spin up a separate Jupyter kernel.  This makes the test fast and
  allows your existing fixtures and monkeypatching to apply to notebook
  code.
* **Selective cell execution** – notebooks often contain explanatory cells,
  plots or intentionally failing examples that you do not want to run in CI.
  A simple directive language embedded in Python comments lets you control
  which cells are included in testing and whether exceptions are expected.
* **Configurable runtime** – options exist to set the NumPy RNG seed, to
  force a non‑interactive matplotlib backend and plotly renderer, to strip
  IPython magics, and to dump the generated test code on failure for easy
  debugging.

The rest of this document describes how notebooks are discovered and
executed, the directive syntax understood by the plugin, the available
configuration surfaces, and guidance on how to integrate the plugin into
your project.

## Notebook discovery

The plugin registers a `pytest_collect_file` hook that recognises any
file ending in `.ipynb`.  When such a file is encountered it is parsed
with [`nbformat`](https://nbformat.readthedocs.io/) and the notebook
cells are processed.  Each notebook yields a single pytest item which
runs the selected cells in order.  Test discovery therefore follows the
usual pytest rules: notebooks in directories named `tests/`, files
matching the `test_*.py` pattern, or any files explicitly passed on the
command line will be collected.  You do not need to pass notebook files
directly to pytest – if they live alongside your Python tests they will
be picked up automatically.

## Directive language

The directive language is embedded in Python comments at the top level of
code cells.  A directive has the general form:

```python
# notebook-test: <flag>=<value>
```

There may be zero to four spaces around the colon, flag name, equals
sign and value.  Flags must appear at most once per cell.  Boolean
directives accept `True` and `False` (case sensitive), while timeout
directives accept numeric seconds.  Unrecognised values or multiple
occurrences of the same flag will produce a test failure at collection
time.  Directives are ignored in markdown cells.

Five directives are currently supported:

### `default-all`

```
# notebook-test: default-all=True|False
```

This sets the default inclusion status for subsequent code cells.  The
plugin maintains a mutable `test_all_cells` variable as it walks through
the notebook.  The initial value of `test_all_cells` comes from the
command line option `--notebook-default-all` (default `True`).

Whenever a cell contains a `default-all` directive the value of
`test_all_cells` is updated for all later cells.  This allows you to
enable or disable execution of entire blocks of cells without annotating
each one individually.  For example, to skip all cells until re‑enabled:

```python
# notebook-test: default-all=False
# this cell and following cells will not be executed

# ... some explanatory or plotting cells ...

# notebook-test: default-all=True
# execution resumes from here
```

### `test-cell`

```
# notebook-test: test-cell=True|False
```

This overrides the current default for the current cell only.  When
absent, the cell inherits the `test_all_cells` value.  When present,
`test-cell=True` forces inclusion regardless of the default and
`test-cell=False` forces exclusion.  A common pattern is to disable
execution by default and then opt in specific cells:

```python
# notebook-test: default-all=False

...  # many cells not executed

# notebook-test: test-cell=True
# code here will run under pytest
```

### `must-raise-exception`

```
# notebook-test: must-raise-exception=True|False
```

By default cells are expected to execute without raising an exception.  If
`must-raise-exception=True` is set on a cell then the entire cell body
is wrapped in a `with pytest.raises(Exception)` context.  Any subclass
of Python’s `Exception` is accepted.  The test will fail if no
exception is raised or if a `BaseException` such as `SystemExit` is
thrown.  When an exception is successfully caught the plugin prints the
exception class name and message to aid debugging.

This flag is not persistent – it applies only to the cell on which it
appears.  An example:

```python
# notebook-test: must-raise-exception=True
raise ValueError("Intentional failure for demonstration")
```

### `notebook-timeout-seconds`

```
# notebook-test: notebook-timeout-seconds=<float>
```

Set a wall-clock timeout (seconds) for the entire notebook.  Enforcement
is delegated to `pytest-timeout`, so that plugin must be installed and
active.  The timeout is a hard deadline across all executed cells.
This directive must appear in the first code cell of the notebook.

### `cell-timeout-seconds`

```
# notebook-test: cell-timeout-seconds=<float>
```

Set a per-cell timeout in seconds.  This overrides any default cell
timeout configured via CLI/ini settings.  Enforcement is delegated to
`pytest-timeout`.

## Runtime prelude and code transformation

Before executing any notebook code the plugin injects a minimal prelude
that imports `pytest` so `pytest.raises` is available for exception
handling.

After the prelude the plugin defines a wrapper function
`run_notebook()` (or `async def run_notebook()` in async mode) into
which the selected cell code is injected.  Each selected cell is
preceded by a marker comment of the form:

```python
## notebook-test notebook=<filename> cell=<index>
```

to make it easy to locate a failing cell in a traceback.  Cells marked
with `must-raise-exception=True` are wrapped in a
`with pytest.raises(Exception)` context manager and will print the
exception type and message.

IPython line magics (lines starting with `%` up to five leading
whitespace characters) are always commented out.  This simple
transformation treats any line beginning with a percent sign as a magic
and prefixes it with `#%`, converting it into a Python comment.  Cell
magics (such as `%%bash`) and shell escapes (`!command`) are not
supported and will result in a test failure.

## Configuration surfaces

The plugin exposes multiple configuration channels.  The order of
precedence is: **command line options** (highest), **pytest.ini or
pyproject settings**, **environment variables**, and then **hard coded
defaults**.

### Command line options

Use `pytest --help` to view all available options.  The most relevant
ones are summarised below.

| Option | Default | Description |
|-------|---------|-------------|
| `--notebook-default-all {true,false}` | `true` | Initial value of the `test_all_cells` flag.  If `false` then cells without an explicit `test-cell` directive will be skipped until `default-all=True` is encountered. |
| `--notebook-keep-generated` | `onfail` | Controls dumping of the generated test script.  `none` means never dump; `onfail` dumps the script into the report upon a test failure; any other string is treated as a path and the script is written there with a filename derived from the notebook name. |
| `--notebook-exec-mode {async,sync}` | `async` | Whether to generate `async def` or `def` for the wrapper.  If `async`, the plugin marks the test item with `pytest.mark.asyncio` if the `pytest-asyncio` plugin is installed.  If `sync`, the code runs synchronously. |
| `--notebook-timeout-seconds` | `none` | Wall-clock timeout for an entire notebook, enforced via `pytest-timeout`. |
| `--notebook-cell-timeout-seconds` | `none` | Default per-cell timeout in seconds, enforced via `pytest-timeout`. |

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

## Report integration and debugging

When a notebook test fails the plugin attaches the generated test script
to the pytest report.  If `--notebook-keep-generated=onfail` (the
default) then a “generated notebook script” section appears in the
report, containing the exact Python code that was executed.  This
mirrors the behaviour of the original harness, which printed the
generated file on failure.

Notebook failures also emit a simplified, cell-focused traceback that
includes only the failing cell's code and the exception message.  This
keeps the output short while still pointing to the relevant cell.

If you supply a directory to `--notebook-keep-generated`, the plugin
writes the generated script to that directory using a name derived from
the notebook filename.  This can be useful for inspecting the code even
when tests pass.  Set `--notebook-keep-generated=none` to disable
generation entirely.

The marker comments inserted into the code (`## notebook-test
notebook=<name> cell=<index>`) allow you to correlate test failures
with the original cell number in the notebook.  Line numbers in
tracebacks also correspond to lines in the generated script when the
script is dumped.

## Limitations and future work

* Timeouts rely on the `pytest-timeout` plugin; install and enable it to
  use `notebook-timeout-seconds` and `cell-timeout-seconds`.
* Only IPython line magics with a single leading `%` are handled.  Cell
  magics (e.g. `%%bash`) and shell escapes (`!`) are not supported and
  will cause a syntax error.
* Notebooks are executed in a single global namespace per item.  They
  are not isolated between cells beyond normal Python scoping.

## Usage summary

1. Install the plugin by adding a Git URL dependency in your
   `pyproject.toml` (for example:
   `pytest-notebook-test @ git+https://github.com/brycehenson/pytest_notebook@main`), then
   install your project as normal.
2. Run pytest.  Any notebooks discovered will be collected as tests.
3. Use inline directives to control which cells run and whether
   exceptions are expected.
4. Adjust runtime options via CLI flags, environment variables or ini
   settings to suit your environment.

## Demo

Run the demo harness to see the notebook-to-pytest flow end-to-end:

```bash
python run_demo.py
```

The demo copies a small set of notebooks into a temporary workspace,
invokes pytest, and reports the outcome for each scenario.  By default it exercises:

* `tests/notebooks/test_simple.ipynb` for a basic pass case.
* `tests/notebooks/test_async_exec_mode.ipynb` to demonstrate async
  execution.
* `tests/notebooks/test_sync_exec_mode.ipynb` to show the sync execution
  path and how to inspect generated scripts.

When a demo uses `--notebook-keep-generated`, the harness prints the
temporary directory so you can open the generated `.py` files and see
the compiled notebook cells.

## Testing

The pytest suite in `tests/test_plugin.py` exercises the plugin using
notebooks under `tests/notebooks/`.  New cases cover:

* async vs. sync execution (`--notebook-exec-mode=sync`)
* generated script retention (`--notebook-keep-generated=none`)

Run the tests as usual:

```bash
pytest
```

To run just a single notebook or a subset by name, use direct paths or
`--notebook-glob`.  For example:

```bash
# Run a single notebook explicitly.
pytest tests/notebooks/example_simple_123.ipynb

# Run every notebook matching a pattern.
pytest tests/notebooks --notebook-glob "test_*.ipynb"
```
