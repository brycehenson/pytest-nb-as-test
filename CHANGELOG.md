# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project follows Semantic Versioning.

## Unreleased

## 0.1.8
- fix notebook-local multiprocessing targets in sync execution by avoiding
  `run_notebook.<locals>` definitions for pickled callables
- harden sync notebook execution by generating an importable helper module for
  top-level notebook function/class definitions and rewriting sync execution to
  import from that module
- isolate sync notebook execution state per run using unique module namespaces
  to reduce cross-test symbol collisions and state leakage
- update uv package manager

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
