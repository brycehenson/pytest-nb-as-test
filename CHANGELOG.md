# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project follows Semantic Versioning.

## Unreleased

## 1.0.0
Version 1.0.0 focuses on making notebook execution behave much more like a real
Jupyter runtime while tightening multiprocessing safety for notebook-defined
targets. It also significantly increases regression and CI coverage, especially
for Windows and Python 3.14, to improve reliability across supported versions.

- improve notebook execution parity with Jupyter by executing generated notebook code at module scope
- execute notebooks with `__name__ == "__main__"` semantics so guarded `main()` blocks run
- preserve multiprocessing compatibility while using `__main__` semantics by aliasing the notebook runtime module during execution
- add clear guardrails for `spawn`/`forkserver` multiprocessing with notebook-defined callables
- harden `spawn`/`forkserver` guardrails by validating callables at class-level multiprocessing entrypoints
  (`multiprocessing.process.BaseProcess.start`, `multiprocessing.pool.Pool.*`,
  and `concurrent.futures.process.ProcessPoolExecutor.submit/map`) to prevent
  import-path bypasses and produce deterministic guardrail errors
- improve directive parsing by rejecting directives indented by more than 4 spaces
- improve directive parsing by allowing trailing inline comments in directive values
- document IPython runtime compatibility limits (magics/shell escapes, `get_ipython`, IPython globals)
- emit pytest warnings when magics/shell escapes are commented out or IPython runtime globals are referenced
- add notebook regression tests for multiprocessing + async cases, guardrail failures, and directive parsing edge cases
- add regression coverage for `if __name__ == "__main__":` notebook execution (`tests/notebooks/test_main.ipynb`)
- add guardrail regression coverage for `multiprocessing.Process`, `get_context("spawn").Process`,
  `from multiprocessing.pool import Pool`, and
  `from concurrent.futures.process import ProcessPoolExecutor`
- expand CI coverage across Python 3.10 to 3.14 and min/latest supported pytest versions
- add a Windows CI test job (Python 3.14, latest supported pytest)
- add a CI smoke test without `pytest-xdist`
- update packaging/project metadata (`MANIFEST.in`, PyPI classifiers, `uv` default groups)
- fix tests/notebooks/test_multiprocessing_local_function.ipynb for py 3.14
- add windows test
- fix a flaky Windows `pytest-xdist` worker crash caused by nested timeout
  integration overwriting `pytest-timeout`'s `item.cancel_timeout` handle;
  preserve/restore the outer timeout cancel callback while arming notebook
  cell/notebook timers so outer test-level timeouts remain cancelable
- add regression coverage for nested timeout-handle preservation in
  `tests/test_timeout.py`


## 0.1.8
- fix notebook-local multiprocessing targets in sync execution by avoiding
  `run_notebook.<locals>` definitions for pickled callables
- harden sync notebook execution by generating an importable helper module for
  top-level notebook function/class definitions and rewriting sync execution to
  import from that module
- isolate sync notebook execution state per run using unique module namespaces
  to reduce cross-test symbol collisions and state leakage
- update uv package manager
- devcontainer:
  - switch to `mcr.microsoft.com/devcontainers/base:debian` and install system
    Python/tooling explicitly to avoid conflicts from preinstalled `pipx` tools
  - remove ad-hoc git tool installs from the Dockerfile and rely on project
    dependency management
- development tooling:
  - add `pre-commit`, `nbstripout-fast`, and `nbdime` to the `dev` dependency
    group in `pyproject.toml`
- update version in `pyproject.toml`

## 0.1.7
-  found error with pytest 7.1.0 and excluded it, broadened pytest
  versions  "pytest>=2.1.0,<9.0.2,!=7.1.0,!=3.2.4,!=2.0.3"
- change default `--notebook-exec-mode` from `async` to `auto` for intelligent execution mode detection
- implement auto-detection of `await` statements to generate async wrappers only when needed, avoiding unnecessary asyncio overhead for synchronous notebooks
- refactor execution path to use universal sync handler that intelligently executes both sync and async code
- stop relying on private pytest traceback internals in notebook failure reporting
- testing:
  - allow selecting a Python interpreter/version for compatibility probe venvs
  - improved `tools/probe_version_compatability.py`,

## 0.1.6
- expand compatible pytest versions
- add probe of version compatability
- fix plotly conftest.py example
- preserve case for `--notebook-keep-generated` paths
- fix `--notebook-exec-mode` error message interpolation
- fix README CI badge link to correct repository
- fix `--notebook-glob` help text to remove stale `--notebook-dir` reference
- fix IPython magic stripping for indented lines
- add test coverage for indented magics
- fix `repr_failure` type annotation to satisfy pyright without private pytest `TerminalRepr` import
- move `tool.coverage.run` and `tool.pytest.ini_options` config into `pyproject.toml`
- accept newer pytest-timeout failure message format in timeout tests
- integrate pytest-asyncio event loop usage for async notebook execution and update docs

## 0.1.5
- update readme with install instructions
- change to real pypi
- fix workflow call
- fix upload url

## 0.1.4
- change notebook directive base to match package name
```
# notebook-test: <flag>=<value>
```
becomes
```
# pytest-nb-as-test: <flag>=<value>
```
- change repository upload url

## 0.1.3
-  no change, testing release
## 0.1.2
-  check pyproject.toml version number is same as tag
## 0.1.1
- no change, testing release

## 0.1.0
- Initial release of the pytest notebook collection and execution plugin.
- Cell directives: `default-all`, `test-cell`, `must-raise-exception`,
  `notebook-timeout-seconds`, `cell-timeout-seconds`.
- CLI options: `--notebook-default-all`, `--notebook-glob`,
  `--notebook-keep-generated`, `--notebook-exec-mode`,
  `--notebook-timeout-seconds`, `--notebook-cell-timeout-seconds`.
- Comprehensive test coverage for collection, directives, execution, and reporting.
