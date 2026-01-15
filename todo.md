resolve this 



Walking down from 8.4.2 to older versions.
PASS: 8.4.2 (46.8 s)
PASS: 7.4.4 (44.0 s)
FAIL: 6.2.5 (25.1 s), rc=1
  stderr:
/tmp/pytest-venv.flw7_6qj/lib/python3.10/site-packages/_pytest/config/__init__.py:318: PluggyTeardownRaisedWarning: A plugin raised an exception during an old-style hookwrapper teardown.
Plugin: helpconfig, Hook: pytest_cmdline_parse
PluginValidationError: Plugin 'pytest_nb_as_test' for hook 'pytest_collect_file'
hookimpl definition: pytest_collect_file(parent: 'pytest.Collector', file_path: 'Path') -> 'pytest.File | None'
Argument(s) {'file_path'} are declared in the hookimpl but can not be found in the hookspec
For more information see https://pluggy.readthedocs.io/en/stable/api_reference.html#pluggy.PluggyTeardownRaisedWarning
  config = pluginmanager.hook.pytest_cmdline_parse(
Traceback (most recent call last):
  File "/opt/pyenv/versions/3.10.19/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/opt/pyenv/versions/3.10.19/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/tmp/pytest-venv.flw7_6qj/lib/python3.10/site-packages/pytest/__main__.py", line 5, in <module>
    raise SystemExit(pytest.console_main())
  File "/tmp/pytest-venv.flw7_6qj/lib/python3.10/site-packages/_pytest/config/__init__.py", line 185, in console_main
    code = main()
  File "/tmp/pytest-venv.flw7_6qj/lib/python3.10/site-packages/_pytest/config/__init__.py", line 143, in main
    config = _prepareconfig(args, plugins)
  File "/tmp/pytest-venv.flw7_6qj/lib/python3.10/site-packages/_pytest/config/__init__.py", line 318, in _prepareconfig
    config = pluginmanager.hook.pytest_cmdline_parse(
  File "/tmp/pytest-venv.flw7_6qj/lib/python3.10/site-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
  File "/tmp/pytest-venv.flw7_6qj/lib/python3.10/site-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
  File "/tmp/pytest-venv.flw7_6qj/lib/python3.10/site-packages/pluggy/_callers.py", line 167, in _multicall
    raise exception
  File "/tmp/pytest-venv.flw7_6qj/lib/python3.10/site-packages/pluggy/_callers.py", line 139, in _multicall
    teardown.throw(exception)
  File "/tmp/pytest-venv.flw7_6qj/lib/python3.10/site-packages/pluggy/_callers.py", line 43, in run_old_style_hookwrapper
    teardown.send(result)
  File "/tmp/pytest-venv.flw7_6qj/lib/python3.10/site-packages/_pytest/helpconfig.py", line 100, in pytest_cmdline_parse
    config: Config = outcome.get_result()
  File "/tmp/pytest-venv.flw7_6qj/lib/python3.10/site-packages/pluggy/_result.py", line 103, in get_result
    raise exc.with_traceback(tb)
  File "/tmp/pytest-venv.flw7_6qj/lib/python3.10/site-packages/pluggy/_callers.py", line 38, in run_old_style_hookwrapper
    res = yield
  File "/tmp/pytest-venv.flw7_6qj/lib/python3.10/site-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
  File "/tmp/pytest-venv.flw7_6qj/lib/python3.10/site-packages/_pytest/config/__init__.py", line 1003, in pytest_cmdline_parse
    self.parse(args)
  File "/tmp/pytest-venv.flw7_6qj/lib/python3.10/site-packages/_pytest/config/__init__.py", line 1283, in parse
    self._preparse(args, addopts=addopts)
  File "/tmp/pytest-venv.flw7_6qj/lib/python3.10/site-packages/_pytest/config/__init__.py", line 1172, in _preparse
    self.pluginmanager.load_setuptools_entrypoints("pytest11")
  File "/tmp/pytest-venv.flw7_6qj/lib/python3.10/site-packages/pluggy/_manager.py", line 417, in load_setuptools_entrypoints
    self.register(plugin, name=ep.name)
  File "/tmp/pytest-venv.flw7_6qj/lib/python3.10/site-packages/_pytest/config/__init__.py", line 436, in register
    ret: Optional[str] = super().register(plugin, name)
  File "/tmp/pytest-venv.flw7_6qj/lib/python3.10/site-packages/pluggy/_manager.py", line 168, in register
    self._verify_hook(hook, hookimpl)
  File "/tmp/pytest-venv.flw7_6qj/lib/python3.10/site-packages/pluggy/_manager.py", line 343, in _verify_hook
    raise PluginValidationError(
pluggy._manager.PluginValidationError: Plugin 'pytest_nb_as_test' for hook 'pytest_collect_file'
hookimpl definition: pytest_collect_file(parent: 'pytest.Collector', file_path: 'Path') -> 'pytest.File | None'
Argument(s) {'file_path'} are declared in the hookimpl but can not be found in the hookspec

Walking up from 8.4.2 to newer versions.
PASS: 9.0.2 (42.6 s)

Last passing version on downward walk: 7.4.4
vscode ➜ /workspaces/pytest_notebooks (pytest_api_fixes) $ 