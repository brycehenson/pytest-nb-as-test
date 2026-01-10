## What your current harness does, in pytest terms

Your file is already a pytest-driven notebook runner, but it is implemented as a normal test module (`test_notebooks.py`) rather than as a pytest plugin.

Core design choices:

* **Execution model**: it does **not** execute a notebook through a Jupyter kernel.
  It parses `.ipynb` with `nbformat`, selects code cells, concatenates their sources into a single generated Python test function, imports that generated module, then runs the function inside pytest. This means “notebook execution” here is really “execute selected cell sources as Python in-process”.

* **Granularity**: one pytest test per notebook, not one per cell. The generated function runs selected cells sequentially, inside one function scope.

* **Selection interface**: per-cell inline comment directives (not cell tags or notebook metadata).

These properties are precisely where existing notebook plugins differ, and where a dedicated pytest plugin version of your harness would be meaningfully distinct.

---

## Proposed implementation as a pytest add-on

### 1) Package structure and plugin registration

Create a distribution, for example `pytest-notebook-test`:

* `pytest_notebook_test/plugin.py` implements hooks.
* `pyproject.toml` registers entry point:

```toml
[project.entry-points.pytest11]
notebook_test = "pytest_notebook_test.plugin"
```

Pytest will auto-load it when installed. ([docs.pytest.org][1])

### 2) Collection: treat `.ipynb` as first-class pytest nodes

Implement a custom collector, analogous to how doctest collection works. Pytest explicitly supports custom file collectors. ([docs.pytest.org][2])

Hook:

* `pytest_addoption(parser)` to add CLI and ini options.
* `pytest_collect_file(file_path, parent)` to collect `.ipynb` into a custom `NotebookFile` node when enabled.

Collector node responsibilities:

* Parse notebook with `nbformat.read`.
* Build the effective per-cell execution plan (your directive logic).
* Create one `NotebookItem` per notebook (or optionally one per “selected cell block”, see options).

### 3) Execution: compile and run generated code without writing a temp file (optional), but preserve debuggability

You have two reasonable execution strategies.

#### Strategy A: keep your current “generate a .py file and import it”

Pros: line numbers and tracebacks are naturally anchored to a real file, easy “print generated code on failure”.
Cons: more filesystem churn, importlib edge cases, temp paths.

This is basically what you do today, just moved behind a collector.

#### Strategy B: generate code in-memory, compile with a synthetic filename, and optionally dump on failure

* Create a single string `source_text` that includes your header plus the function definition.
* Call `compile(source_text, filename=f"<notebook {path}>", mode="exec")`.
* `exec` into an isolated globals dict, extract `test_<notebook_name>`.

To keep your current debug ergonomics:

* Always keep `source_text` on the `NotebookItem`.
* On failure, attach it to the report and optionally dump it to disk via `--notebook-test-dump-code`.

### 4) Async integration: handle `async def` cells correctly

Your current generator always writes `async def test_<name>():` and then `await test_function()`, relying on `pytest.mark.asyncio` to run the coroutine.

As a plugin, you should not assume users have `pytest-asyncio`, but you can support both:

* If `pytest-asyncio` is installed, mark the item with `pytest.mark.asyncio`.
* Otherwise, run via `asyncio.run` inside `NotebookItem.runtest()`.

That yields a self-contained plugin. If a repo already uses `pytest-asyncio`, your behaviour matches their event loop policy.

### 5) Recreate your directive semantics, but formalise them

Your directives are:

* `# notebook-test: default-all=<True|False>`
  Sets the default inclusion policy for subsequent code cells.

* `# notebook-test: test-cell=<True|False>`
  Overrides inclusion for that cell only.

* `# notebook-test: must-raise-exception=<True|False>`
  If true, wrap that cell in `with pytest.raises(Exception)`.

You can keep exactly this syntax, since it is user-facing and already deployed.

Implementation details to match your current semantics:

* Parse directives only in **code** cells.
* Allow at most one assignment per directive per cell.
* Enforce literal `True` or `False` only.
* Update a running `test_all_cells` state as you traverse cells, since `default-all` is stateful.

### 6) Define a plugin-level interface: CLI, ini, and per-notebook overrides

Below is a concrete interface proposal that matches your current capabilities and makes them reproducible.

#### CLI options

* `--notebook-test`
  Enable collection of notebooks. (Without it, do nothing, to avoid surprising users.)

* `--notebook-test-path=<glob or path>` (repeatable)
  Additional search roots or globs, similar to your `docs/example_notebooks/**`.

* `--notebook-test-envvar=NOTEBOOK_DIR_TO_TEST`
  Default `NOTEBOOK_DIR_TO_TEST`, preserves your current environment override behaviour.

* `--notebook-test-default-all=<true|false>`
  Default inclusion if no directives set it. Your current default is True.

* `--notebook-test-strip-magics=<true|false>`
  Default true, implement your `re.sub(r"^ {0,5}%", "#%", ...)`.

* `--notebook-test-seed=<int>`
  Default 42, applies `np.random.seed(seed)`.

* `--notebook-test-mpl-backend=<str>`
  Default `Agg`, sets `matplotlib.use(...)`.

* `--notebook-test-plotly-renderer=<str>`
  Default `jpg`, sets `pio.renderers.default = ...`.

* `--notebook-test-dump-code=<never|onfail|always>`
  Mirrors your failure printing, but also supports persisting code for CI artefacts.

* `--notebook-test-timeout=<seconds>` (optional, per notebook)
  A hard timeout around the entire generated test, implemented via `signal` on Unix or `pytest-timeout` integration.

* `--notebook-test-qcs-random=<true|false>`
  Default true in your environment, calls your `set_global_qcs_to_random_name()` before execution.

#### Ini options

All CLI options should have ini equivalents:

* `notebook_test_enabled`
* `notebook_test_paths`
* `notebook_test_default_all`
* `notebook_test_strip_magics`
* `notebook_test_seed`
* `notebook_test_mpl_backend`
* `notebook_test_plotly_renderer`
* `notebook_test_dump_code`
* `notebook_test_qcs_random`
* `notebook_test_timeout`

#### Notebook metadata overrides (optional)

You currently avoid notebook metadata except for standard kernelspec fields, but as a plugin you can allow opt-in overrides under a dedicated namespace, for example `metadata["notebook_test"]`:

* `enabled: bool`
* `default_all: bool`
* `seed: int`
* `timeout: int`
* `dump_code: str`

This is a direct analogue to how other notebook plugins use dedicated metadata namespaces for behaviour. ([pytest-notebook][3])

### 7) Reporting and failure UX

Implement `pytest_runtest_makereport(item, call)`:

* If failure and `dump_code` is enabled, attach the generated source text in the report “longrepr”.
* Optionally include cell index annotations, you already prefix each cell with `## notebook-test notebook=<name> cell=<ii>`.

This gives you deterministic, CI-friendly diagnostics without relying on `print`.

---

## Survey of existing pytest notebook runners, and why they do not match your approach

### nbval

**What it does**: re-runs notebooks and compares the outputs produced by execution against the outputs stored in the `.ipynb` file. It has a strict mode (`--nbval`) and a lax mode (`--nbval-lax`) that only checks outputs for specially marked cells, and it supports regex-based output sanitising. ([Nbval][4])

**Limitations relative to your harness-plus-plugin proposal**:

* **Different contract**: nbval’s main value is output regression against stored outputs, which implies you must commit outputs and keep them stable. Your harness is primarily an **execution sanity test** over selected cells, not an output-regression framework.
* **Selection mechanism mismatch**: nbval’s behaviour is controlled via its flags and tag conventions, not via an inline directive language that can flip “default-all” statefully through a notebook. ([Nbval][4])
* **Kernel dependency**: nbval executes through a Jupyter kernel and captures iopub messages, which increases fidelity but also increases overhead and introduces kernel-level nondeterminism modes that your in-process execution avoids. ([GitHub][5])
* **Exception semantics**: nbval supports “cell raises exception” patterns, but its primary model is still “cells are tests”, whereas your approach is “notebook is one test with a controlled subset and controlled exception expectations”.

### nbmake

**What it does**: executes notebooks with `nbclient` under pytest, mainly to ensure they run without errors, with features like per-cell timeout (`--nbmake-timeout`), forcing a kernel (`--nbmake-kernel`), allowing errors at notebook metadata level, and cell tags like `raises-exception` and `skip-execution`. ([GitHub][6])

**Limitations relative to your proposal**:

* **All-cells default**: nbmake is oriented around executing the notebook top-to-bottom, skipping only via explicit tags like `skip-execution`. Your directive design supports **stateful default policies** (flip default-all midway) and **opt-in cells** without touching tags. ([GitHub][6])
* **Kernel execution**: like nbval, nbmake’s kernel model is heavier than your “execute as Python” model, and it preserves notebook semantics that you explicitly choose to discard (for example magics, rich display machinery). ([GitHub][6])
* **Fixture integration**: nbmake is “notebook as artefact”, it does not naturally make the notebook code behave like a normal pytest test module with ordinary imports and process-level fixtures. Your approach runs in the same interpreter, so it can use the same dependency injection patterns and monkeypatching mechanisms.

### pytest-notebook

**What it does**: regression testing and regenerating notebooks, with diffing via nbdime, configuration at CLI, ini, and notebook metadata levels, plus metadata-driven skip and diff-ignore or diff-replace rules under a dedicated namespace. ([pytest-notebook][7])

**Limitations relative to your proposal**:

* **Regression focus**: it is built around diffing “expected notebook” versus “obtained notebook”. Your harness is a selective execution harness, not a notebook-output diff system.
* **Metadata-first**: its customisation model is notebook metadata oriented (for example `nbreg.skip`, diff paths). Your harness’s control plane is a line-level directive language embedded directly in code cells. ([pytest-notebook][3])
* **Selection granularity**: it does not target “build a single Python test function from selected cells”, it targets “execute notebook then diff”.

### pytest-ipynb2 (and older pytest-ipynb)

**What it does**: collects tests defined inside notebooks, typically via an `%%ipytest` cell magic, and then runs those tests, executing prior cells to set up state. ([musicalninjadad.github.io][8])

**Limitations relative to your proposal**:

* **Tests must be written as pytest tests in the notebook**, not just ordinary code cells that you want to execute as an integration smoke test.
* **Side effects during collection**: it explicitly warns that executing prior cells means side effects occur as part of collection-like behaviour. ([musicalninjadad.github.io][8])
* **Not aimed at “subset-of-cells smoke testing”**: your directive interface is tailored to documentation notebooks where only some cells should be executed in CI.

### pytest-jupyter / pytest-jupyter-client

**What it does**: provides fixtures and helpers to test Jupyter servers and kernel sessions, it is infrastructure for Jupyter components, not a notebook runner. It also notes async-runner compatibility constraints (pytest-tornasync versus pytest-asyncio). ([GitHub][9])

**Limitations relative to your proposal**:

* It does not provide notebook discovery, cell selection, or notebook execution orchestration.
* Its scope is lower-level, you would build a notebook runner on top of it, but it would not replace your directive-driven design.

---

## Why your “pytest plugin version” is plausibly better for your use case

Your harness is tuned for a particular niche:

* Documentation or example notebooks that are “mostly Python”, but contain cells that should be skipped in CI, or cells that intentionally demonstrate failures.
* A desire to keep notebook control flow close to code, via visible comment directives, rather than hidden metadata tags.
* Running in the same interpreter as pytest, so ordinary pytest tooling applies, including monkeypatching, environment control, and repository-specific global initialisation.

None of nbval, nbmake, or pytest-notebook reproduce that exact combination of:

* **stateful default-all selection**,
* **inline directive syntax**,
* **single-test-per-notebook execution in-process**,
* **simple “must raise” wrapping**,
* **built-in rendering and RNG stabilisation prelude**. ([GitHub][6])

If you want, I can also sketch the concrete `NotebookFile` and `NotebookItem` classes and the hook implementations, but the above is the functional spec that would let you recreate your current behaviour as a clean pytest extension.

[1]: https://docs.pytest.org/en/stable/how-to/plugins.html?utm_source=chatgpt.com "How to install and use plugins"
[2]: https://docs.pytest.org/en/stable/example/markers.html?utm_source=chatgpt.com "Working with custom markers"
[3]: https://pytest-notebook.readthedocs.io/en/latest/literal_includes/nb_metadata_schema.html "Notebook/Cell Metadata Schema — pytest-notebook 0.10.0 documentation"
[4]: https://nbval.readthedocs.io/ "IPython Notebook Validation for py.test - Documentation — nbval 0.11.0 documentation"
[5]: https://github.com/computationalmodelling/nbval "GitHub - computationalmodelling/nbval: A py.test plugin to validate Jupyter notebooks"
[6]: https://github.com/treebeardtech/nbmake "GitHub - treebeardtech/nbmake:  Pytest plugin for testing notebooks"
[7]: https://pytest-notebook.readthedocs.io/ "pytest-notebook — pytest-notebook 0.10.0 documentation"
[8]: https://musicalninjadad.github.io/pytest-ipynb2/ "pytest-ipynb2"
[9]: https://github.com/jupyter-server/pytest-jupyter "GitHub - jupyter-server/pytest-jupyter: A pytest plugin for testing Jupyter core libraries and extensions."
