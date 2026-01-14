# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project follows Semantic Versioning.

## Unreleased
- update readme with install instructions

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
