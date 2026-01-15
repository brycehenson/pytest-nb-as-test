# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project follows Semantic Versioning.

## Unreleased
- expand compatible versions
- add probe of version compatability
- fix plotly conftest.py example
- preserve case for `--notebook-keep-generated` paths
- fix `--notebook-exec-mode` error message interpolation
- fix README CI badge link to correct repository
- fix `--notebook-glob` help text to remove stale `--notebook-dir` reference
- fix IPython magic stripping for indented lines
- add test coverage for indented magics
- avoid private pytest `TerminalRepr` import in item repr annotation
- move `tool.coverage.run` and `tool.pytest.ini_options` config into `pyproject.toml`
- accept newer pytest-timeout failure message format in timeout tests

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
