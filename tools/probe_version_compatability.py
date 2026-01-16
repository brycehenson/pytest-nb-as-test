#!/usr/bin/env python3
"""Probe pytest compatibility across multiple version selections.

Features:
- Walk down from current version to older versions, then up if needed.
- With --walk-major-then-refine: probe major versions first, then refine patch boundary
  when a failure is found.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass(frozen=True)
class ProbeResult:
    """Result of probing a single dependency version."""

    version: str
    passed: bool
    returncode: int
    duration_s: float
    venv_dir: Path
    stdout: str
    stderr: str


@dataclass(frozen=True)
class PreparedEnvironment:
    """A venv ready for pytest execution after setup stage.

    Attributes:
        version: The version string being tested.
        venv_dir: Path to the prepared virtual environment.
        python_exe: Path to the python executable in the venv.
        setup_passed: Whether setup completed successfully.
        setup_error: Error message if setup failed, else None.
        install_stderr: Combined stderr from all installation steps.

    Example:
        env = PreparedEnvironment(
            version="8.0.0",
            venv_dir=Path("/tmp/venv"),
            python_exe=Path("/tmp/venv/bin/python"),
            setup_passed=True,
            setup_error=None,
            install_stderr=""
        )
    """

    version: str
    """Version string being tested."""
    venv_dir: Path
    """Path to the prepared virtual environment."""
    python_exe: Path
    """Path to the python executable in the venv."""
    setup_passed: bool
    """Whether setup completed successfully."""
    setup_error: Optional[str]
    """Error message if setup failed, else None."""
    install_stderr: str
    """Combined stderr from all installation steps."""


def _run(
    cmd: Sequence[str],
    cwd: Path,
    env: dict[str, str],
    *,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command and capture text output."""
    return subprocess.run(
        list(cmd),
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def _venv_python(venv_dir: Path) -> Path:
    """Return the venv python executable path."""
    if platform.system().lower().startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _pyenv_root() -> Optional[str]:
    """Return the pyenv root directory if it is discoverable.

    This prefers the PYENV_ROOT environment variable, then falls back to the
    common devcontainer install location at /opt/pyenv if it exists.

    Returns:
        The pyenv root path, or None if not found.

    Example:
        root = _pyenv_root()
    """
    env_root: Optional[str] = os.environ.get("PYENV_ROOT")
    if env_root:
        return env_root
    default_root: Path = Path("/opt/pyenv")
    if default_root.exists():
        return str(default_root)
    return None


def _find_pyenv_executable() -> Optional[str]:
    """Return a resolved pyenv executable path if available, else None.

    Search order:
      1) pyenv on PATH
      2) $PYENV_ROOT/bin/pyenv
      3) /opt/pyenv/bin/pyenv (if present)

    Returns:
        Absolute path to pyenv if found, else None.

    Example:
        pyenv_exe = _find_pyenv_executable()
    """
    on_path: Optional[str] = shutil.which("pyenv")
    if on_path is not None:
        return on_path

    root: Optional[str] = _pyenv_root()
    if root is None:
        return None

    candidate: Path = Path(root) / "bin" / "pyenv"
    if candidate.exists():
        return str(candidate)
    return None


def _pyenv_command() -> list[str]:
    """Return the pyenv command as a list suitable for subprocess calls.

    Returns:
        A single element list containing the resolved pyenv executable path.

    Raises:
        RuntimeError: If pyenv cannot be found.

    Example:
        cmd = _pyenv_command()
    """
    exe: Optional[str] = _find_pyenv_executable()
    if exe is None:
        raise RuntimeError("pyenv is not available on PATH and PYENV_ROOT is not set.")
    return [exe]


def _pyenv_env(*, pyenv_version: Optional[str] = None) -> dict[str, str]:
    """Return an environment dictionary suitable for pyenv operations.

    Args:
        pyenv_version: If provided, set PYENV_VERSION to select the interpreter.

    Returns:
        Environment dict for subprocess execution.

    Example:
        env = _pyenv_env(pyenv_version="3.9.18")
    """
    env: dict[str, str] = dict(os.environ)

    root: Optional[str] = _pyenv_root()
    if root is not None:
        env["PYENV_ROOT"] = root
        bin_dir: str = str(Path(root) / "bin")
        shims_dir: str = str(Path(root) / "shims")
        path_sep: str = os.pathsep
        env["PATH"] = f"{bin_dir}{path_sep}{shims_dir}{path_sep}{env.get('PATH', '')}"

    if pyenv_version is not None:
        env["PYENV_VERSION"] = pyenv_version

    return env


def _pyenv_latest_patch(version: str) -> str:
    """Resolve X.Y to the highest available X.Y.Z version in pyenv.

    Args:
        version: A string in X.Y format.

    Returns:
        Resolved X.Y.Z string.

    Raises:
        ValueError: If version is not in X.Y format.
        RuntimeError: If no matching patch versions are available.

    Example:
        resolved = _pyenv_latest_patch("3.9")
    """
    if re.match(r"^\d+\.\d+$", version) is None:
        raise ValueError(f"Expected X.Y format (got {version!r}).")

    pyenv_cmd: list[str] = _pyenv_command()

    proc: subprocess.CompletedProcess[str] = _run(
        [*pyenv_cmd, "install", "--list"],
        cwd=Path.cwd(),
        env=_pyenv_env(),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to query pyenv install list.\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}\n"
        )

    mm: str = version
    pat: re.Pattern[str] = re.compile(rf"^\s*{re.escape(mm)}\.(?P<patch>\d+)\s*$")
    best_patch: Optional[int] = None
    for line in proc.stdout.splitlines():
        m: Optional[re.Match[str]] = pat.match(line)
        if m is None:
            continue
        patch_i: int = int(m.group("patch"))
        if best_patch is None or patch_i > best_patch:
            best_patch = patch_i

    if best_patch is None:
        raise RuntimeError(f"No pyenv install candidates found for {mm}.")
    return f"{mm}.{best_patch}"


def _pyenv_install(version: str) -> None:
    """Ensure a pyenv-managed Python version is installed.

    Args:
        version: A string in X.Y.Z format.

    Raises:
        RuntimeError: If installation fails.

    Example:
        _pyenv_install("3.9.18")
    """
    proc: subprocess.CompletedProcess[str] = _run(
        ["whoami"],
        cwd=Path.cwd(),
        env=_pyenv_env(),
        check=False,
    )
    print(f"user : {proc.stdout}")

    pyenv_cmd: list[str] = _pyenv_command()
    print("pyenv installing")
    combined_command = [*pyenv_cmd, "install", "-s", version]
    print(" ".join(combined_command))
    proc: subprocess.CompletedProcess[str] = _run(
        combined_command,
        cwd=Path.cwd(),
        env=_pyenv_env(),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"pyenv install failed for {version}.\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}\n"
        )

    _ = _run(
        [*pyenv_cmd, "rehash"],
        cwd=Path.cwd(),
        env=_pyenv_env(),
        check=False,
    )


def _pyenv_python(version: str) -> list[str]:
    """Return the resolved python executable path for a pyenv version.

    Args:
        version: A string in X.Y.Z format.

    Returns:
        A single element list containing the python executable path.

    Raises:
        RuntimeError: If pyenv cannot resolve the python path.

    Example:
        python_cmd = _pyenv_python("3.9.18")
    """
    pyenv_cmd: list[str] = _pyenv_command()
    proc: subprocess.CompletedProcess[str] = _run(
        [*pyenv_cmd, "which", "python"],
        cwd=Path.cwd(),
        env=_pyenv_env(pyenv_version=version),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"pyenv could not resolve python for {version}.\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}\n"
        )

    resolved: str = proc.stdout.strip().splitlines()[0].strip()
    if not resolved:
        raise RuntimeError(f"pyenv returned an empty python path for {version}.")
    return [resolved]


def _interpreter_version(python_exe: str) -> Optional[str]:
    """Return X.Y.Z version string for a python executable, else None.

    Args:
        python_exe: Path to a python executable.

    Returns:
        Version string in X.Y.Z format, or None on failure.

    Example:
        v = _interpreter_version("/usr/bin/python3.12")
    """
    proc: subprocess.CompletedProcess[str] = _run(
        [
            python_exe,
            "-c",
            (
                "import sys; "
                "print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
            ),
        ],
        cwd=Path.cwd(),
        env=os.environ.copy(),
        check=False,
    )
    if proc.returncode != 0:
        return None
    out: str = proc.stdout.strip()
    if re.match(r"^\d+\.\d+\.\d+$", out) is None:
        return None
    return out


def _resolve_python_cmd(
    python_exe: Optional[str], python_version: Optional[str]
) -> list[str]:
    """Resolve the python command used to create new venvs.

    Args:
        python_exe: Explicit python executable path or name on PATH.
        python_version: Python version string like "3.9" or "3.9.18".

    Returns:
        Command list to invoke the requested Python interpreter.

    Raises:
        RuntimeError: When a matching Python interpreter cannot be found.
        ValueError: When the python_version format is invalid.

    Example:
        python_cmd = _resolve_python_cmd(None, "3.9")
    """
    if python_exe:
        resolved = shutil.which(python_exe)
        if resolved is None:
            raise RuntimeError(f"Python executable {python_exe!r} not found on PATH.")
        return [resolved]

    if python_version:
        match: Optional[re.Match[str]] = re.match(
            r"^(?P<major>\d+)\.(?P<minor>\d+)(?:\.(?P<patch>\d+))?$",
            python_version,
        )
        if match is None:
            raise ValueError(
                "python_version must be in 'X.Y' or 'X.Y.Z' format "
                f"(got {python_version!r})."
            )
        major: str = match.group("major")
        minor: str = match.group("minor")
        patch: Optional[str] = match.group("patch")

        major_minor: str = f"{major}.{minor}"

        candidate: str = f"python{major_minor}"
        resolved_candidate: Optional[str] = shutil.which(candidate)
        if resolved_candidate is not None:
            if patch is None:
                return [resolved_candidate]

            want: str = f"{major_minor}.{patch}"
            got: Optional[str] = _interpreter_version(resolved_candidate)
            if got == want:
                return [resolved_candidate]

        if platform.system().lower().startswith("win"):
            launcher = shutil.which("py")
            if launcher is not None:
                return [launcher, f"-{major}.{minor}"]

        pyenv_exe: Optional[str] = _find_pyenv_executable()
        if pyenv_exe is not None:
            resolved_version: str
            if patch is None:
                resolved_version = _pyenv_latest_patch(major_minor)
            else:
                resolved_version = f"{major_minor}.{patch}"

            _pyenv_install(resolved_version)
            return _pyenv_python(resolved_version)

        if patch is None:
            raise RuntimeError(
                f"Could not find a Python {major_minor} interpreter on PATH."
            )
        raise RuntimeError(
            f"Could not find a Python {major_minor}.{patch} interpreter on PATH."
        )

    return [sys.executable]


def _make_venv(venv_dir: Path, python_cmd: Sequence[str]) -> None:
    """Create a new virtual environment using the requested interpreter.

    Args:
        venv_dir: Target venv directory path.
        python_cmd: Command list for the Python interpreter to use.

    Raises:
        RuntimeError: If the venv creation command fails.

    Example:
        _make_venv(Path("venv"), ["python3.9"])
    """
    proc = _run(
        [*list(python_cmd), "-m", "venv", str(venv_dir)],
        cwd=Path.cwd(),
        env=os.environ.copy(),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to create venv.\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}\n"
        )


def _base_env_for_venv(venv_dir: Path) -> dict[str, str]:
    """Construct an environment dict that prioritises the venv."""
    env: dict[str, str] = dict(os.environ)
    env["VIRTUAL_ENV"] = str(venv_dir)

    bin_dir: Path = venv_dir / (
        "Scripts" if platform.system().lower().startswith("win") else "bin"
    )
    path_sep: str = os.pathsep
    env["PATH"] = f"{bin_dir}{path_sep}{env.get('PATH', '')}"

    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PIP_NO_INPUT"] = "1"
    return env


def _current_installed_version(dist_name: str) -> Optional[str]:
    """Return the installed version of a dist if available in this interpreter, else None."""
    try:
        from importlib import metadata as importlib_metadata  # py312+

        return importlib_metadata.version(dist_name)
    except Exception:
        return None


def _available_versions_via_pip_index(dist_name: str) -> list[str]:
    """Return versions for a dist via pip index, newest first.

    Tries JSON output first, falls back to parsing human output.
    """
    cmd_json: list[str] = [
        sys.executable,
        "-m",
        "pip",
        "index",
        "versions",
        dist_name,
        "--json",
    ]
    proc_json: subprocess.CompletedProcess[str] = _run(
        cmd_json, cwd=Path.cwd(), env=os.environ.copy(), check=False
    )
    if proc_json.returncode == 0:
        try:
            payload: object = json.loads(proc_json.stdout)
            if isinstance(payload, dict):
                versions_obj: object = payload.get("versions")
                if isinstance(versions_obj, list) and all(
                    isinstance(v, str) for v in versions_obj
                ):
                    return list(versions_obj)
        except Exception:
            pass

    cmd_text: list[str] = [sys.executable, "-m", "pip", "index", "versions", dist_name]
    proc_text: subprocess.CompletedProcess[str] = _run(
        cmd_text, cwd=Path.cwd(), env=os.environ.copy(), check=False
    )
    if proc_text.returncode != 0:
        raise RuntimeError(
            "Failed to obtain versions via `pip index versions`.\n"
            f"stdout:\n{proc_text.stdout}\n\nstderr:\n{proc_text.stderr}\n"
        )

    m: Optional[re.Match[str]] = re.search(
        r"^Available versions:\s*(.+)$", proc_text.stdout, flags=re.MULTILINE
    )
    if m is None:
        raise RuntimeError(
            "Could not parse `pip index versions` output.\n"
            f"stdout:\n{proc_text.stdout}\n\nstderr:\n{proc_text.stderr}\n"
        )

    versions_line: str = m.group(1).strip()
    versions: list[str] = [v.strip() for v in versions_line.split(",") if v.strip()]
    if not versions:
        raise RuntimeError(f"Parsed no versions from line: {versions_line!r}")
    return versions


def _latest_per_major(versions: Sequence[str]) -> list[str]:
    """Return only the newest version for each major (newest-first input).

    Args:
        versions: Versions ordered newest-first.

    Returns:
        List of versions containing only the first occurrence per major.

    Example:
        _latest_per_major(["9.0.2", "9.0.1", "8.4.2", "8.3.0"]) == ["9.0.2", "8.4.2"]
    """
    selected: list[str] = []
    seen: set[str] = set()
    for version in versions:
        match: Optional[re.Match[str]] = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
        if match is None:
            continue
        major: str = match.group(1)
        if major in seen:
            continue
        seen.add(major)
        selected.append(version)
    return selected


def _versions_for_major(versions: Sequence[str], major: str) -> list[str]:
    """Return all versions matching a given major version string.

    Args:
        versions: Versions to filter (assumed to be ordered newest-first).
        major: Major version number as a string (e.g., "8").

    Returns:
        List of matching versions ordered newest-first.

    Example:
        _versions_for_major(["8.4.2", "8.4.1", "8.3.0", "7.2.0"], "8") == ["8.4.2", "8.4.1", "8.3.0"]
    """
    matched: list[str] = []
    for version in versions:
        match: Optional[re.Match[str]] = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
        if match is None:
            continue
        if match.group(1) == major:
            matched.append(version)
    return matched


def _filter_excluded_versions(
    versions: Sequence[str], excluded: Sequence[str]
) -> list[str]:
    """Filter out excluded versions from available versions.

    Args:
        versions: All available versions (assumed ordered newest-first).
        excluded: Version strings to exclude (exact matches only).

    Returns:
        Filtered list with excluded versions removed.

    Example:
        _filter_excluded_versions(["7.4.4", "7.1.0", "6.2.5"], ["7.1.0"]) == ["7.4.4", "6.2.5"]
    """
    excluded_set: set[str] = set(excluded)
    return [v for v in versions if v not in excluded_set]


def _setup_pytest_environment(
    *,
    project_root: Path,
    version: str,
    dist_name: str,
    extra_install: Sequence[str],
    python_cmd: Sequence[str],
) -> PreparedEnvironment:
    """Create and configure a virtual environment for testing.

    Performs all setup steps: venv creation, pip upgrades, and dependency installation.
    Does NOT run pytest.

    Args:
        project_root: Root directory of the project to test.
        version: Version string of dist_name to install (e.g., "8.4.2").
        dist_name: Distribution name to probe (e.g., pytest, nbformat).
        extra_install: Additional packages to install after the main dist package.
        python_cmd: Command list for the Python interpreter to use for venv creation.

    Returns:
        PreparedEnvironment with venv_dir, python_exe, and setup status.

    Example:
        env = _setup_pytest_environment(
            project_root=Path("."),
            version="8.0.0",
            dist_name="pytest",
            extra_install=["matplotlib"],
            python_cmd=["python3.10"]
        )
    """
    venv_dir_str: str = tempfile.mkdtemp(prefix="pytest-venv.")
    venv_dir: Path = Path(venv_dir_str)
    vpy: Path = _venv_python(venv_dir)
    env: dict[str, str] = _base_env_for_venv(venv_dir)

    install_stderr: list[str] = []

    try:
        _make_venv(venv_dir, python_cmd)

        # Upgrade pip
        proc: subprocess.CompletedProcess[str] = _run(
            [str(vpy), "-m", "pip", "install", "-U", "pip"],
            cwd=project_root,
            env=env,
            check=False,
        )
        install_stderr.append(proc.stderr)
        if proc.returncode != 0:
            return PreparedEnvironment(
                version=version,
                venv_dir=venv_dir,
                python_exe=vpy,
                setup_passed=False,
                setup_error=f"pip upgrade failed: {proc.stderr}",
                install_stderr="".join(install_stderr),
            )

        # Install main dist package with extras
        proc = _run(
            [str(vpy), "-m", "pip", "install", "-U", f"{dist_name}=={version}"],
            cwd=project_root,
            env=env,
            check=False,
        )
        install_stderr.append(proc.stderr)
        if proc.returncode != 0:
            return PreparedEnvironment(
                version=version,
                venv_dir=venv_dir,
                python_exe=vpy,
                setup_passed=False,
                setup_error=f"main package install failed: {proc.stderr}",
                install_stderr="".join(install_stderr),
            )

        # Install extra packages
        if extra_install:
            proc = _run(
                [str(vpy), "-m", "pip", "install", *list(extra_install)],
                cwd=project_root,
                env=env,
                check=False,
            )
            install_stderr.append(proc.stderr)
            if proc.returncode != 0:
                return PreparedEnvironment(
                    version=version,
                    venv_dir=venv_dir,
                    python_exe=vpy,
                    setup_passed=False,
                    setup_error=f"extra package install failed: {proc.stderr}",
                    install_stderr="".join(install_stderr),
                )

        # Install project in editable mode
        proc = _run(
            [str(vpy), "-m", "pip", "install", "-e", "."],
            cwd=project_root,
            env=env,
            check=False,
        )
        install_stderr.append(proc.stderr)
        if proc.returncode != 0:
            return PreparedEnvironment(
                version=version,
                venv_dir=venv_dir,
                python_exe=vpy,
                setup_passed=False,
                setup_error=f"project install failed: {proc.stderr}",
                install_stderr="".join(install_stderr),
            )

        time.sleep(0.5)

        return PreparedEnvironment(
            version=version,
            venv_dir=venv_dir,
            python_exe=vpy,
            setup_passed=True,
            setup_error=None,
            install_stderr="".join(install_stderr),
        )

    except Exception as e:
        return PreparedEnvironment(
            version=version,
            venv_dir=venv_dir,
            python_exe=vpy,
            setup_passed=False,
            setup_error=f"Unexpected error during setup: {e}",
            install_stderr="".join(install_stderr),
        )


def _run_pytest_in_environment(
    *,
    prepared_env: PreparedEnvironment,
    project_root: Path,
    pytest_cmd: Sequence[str],
    keep_failed_venv: bool,
    verbose: int = 0,
    worker_id: int = -1,
) -> ProbeResult:
    """Run pytest in a prepared environment.

    Assumes the environment has already been set up via _setup_pytest_environment.
    Cleans up the venv after execution unless setup failed or keep_failed_venv is True.

    Args:
        prepared_env: A PreparedEnvironment from _setup_pytest_environment.
        project_root: Root directory of the project to test.
        pytest_cmd: Pytest command to run, excluding the python executable prefix.
        keep_failed_venv: If True, keep the venv directory when the probe fails.
        verbose: Verbosity level for logging.
        worker_id: Worker ID for logging purposes.

    Returns:
        ProbeResult containing pass/fail, outputs, and timing.

    Example:
        result = _run_pytest_in_environment(
            prepared_env=env,
            project_root=Path("."),
            pytest_cmd=["-m", "pytest"],
            keep_failed_venv=False,
            verbose=2,
            worker_id=0
        )
    """
    t0: float = time.time()
    venv_dir: Path = prepared_env.venv_dir
    vpy: Path = prepared_env.python_exe
    env: dict[str, str] = _base_env_for_venv(venv_dir)

    stdout_all: list[str] = []
    stderr_all: list[str] = []
    passed: bool = False
    returncode: int = 1
    worker_prefix: str = (
        f"[Exec Worker {worker_id}] " if worker_id >= 0 and verbose >= 2 else ""
    )

    try:
        # If setup failed, return immediately with setup error
        if not prepared_env.setup_passed:
            if verbose >= 2:
                print(f"{worker_prefix}Setup failed for version {prepared_env.version}")
            return ProbeResult(
                version=prepared_env.version,
                passed=False,
                returncode=1,
                duration_s=time.time() - t0,
                venv_dir=venv_dir,
                stdout="",
                stderr=prepared_env.setup_error or "Unknown setup error",
            )

        cache_dir: str = f"/tmp/pytest_cache/{prepared_env.version}"
        proc: subprocess.CompletedProcess[str] = _run(
            [str(vpy), *list(pytest_cmd), "-o", f"cache_dir={cache_dir}"],
            cwd=project_root,
            env=env,
            check=False,
        )
        stdout_all.append(proc.stdout)
        stderr_all.append(proc.stderr)

        returncode = proc.returncode
        passed = proc.returncode == 0
        return ProbeResult(
            version=prepared_env.version,
            passed=passed,
            returncode=returncode,
            duration_s=time.time() - t0,
            venv_dir=venv_dir,
            stdout="".join(stdout_all),
            stderr="".join(stderr_all) + prepared_env.install_stderr,
        )
    finally:
        if passed or not keep_failed_venv:
            shutil.rmtree(venv_dir, ignore_errors=True)


def probe_pytest_version(
    *,
    project_root: Path,
    pytest_version: str,
    dist_name: str,
    keep_failed_venv: bool,
    extra_install: Sequence[str],
    pytest_cmd: Sequence[str],
    python_cmd: Sequence[str],
) -> ProbeResult:
    """Create an isolated venv, install dependencies, run pytest, then clean up.

    This is the legacy monolithic interface. For new code, use _setup_pytest_environment
    and _run_pytest_in_environment separately to enable pipelining.

    Args:
        project_root: Root directory of the project to test, used as cwd for installs and pytest.
        pytest_version: Exact version string to install, for example "8.4.2".
        dist_name: Distribution name to probe (e.g., pytest, nbformat).
        keep_failed_venv: If True, keep the venv directory when the probe fails.
        extra_install: Additional packages to install after the main dist package.
        pytest_cmd: Pytest command to run, excluding the python executable prefix.
        python_cmd: Command list for the Python interpreter to use for venv creation.

    Returns:
        ProbeResult containing pass/fail, outputs, and timing.

    Example:
        result = probe_pytest_version(
            project_root=Path("."),
            pytest_version="8.0.0",
            dist_name="pytest",
            keep_failed_venv=False,
            extra_install=["matplotlib"],
            pytest_cmd=["-m", "pytest"],
            python_cmd=["python3.10"]
        )
    """
    prepared_env: PreparedEnvironment = _setup_pytest_environment(
        project_root=project_root,
        version=pytest_version,
        dist_name=dist_name,
        extra_install=extra_install,
        python_cmd=python_cmd,
    )
    return _run_pytest_in_environment(
        prepared_env=prepared_env,
        project_root=project_root,
        pytest_cmd=pytest_cmd,
        keep_failed_venv=keep_failed_venv,
    )


def _pipelined_parallel_probe(
    *,
    versions: Sequence[str],
    project_root: Path,
    dist_name: str,
    max_workers: int,
    stop_on_first_fail: bool,
    keep_failed_venv: bool,
    extra_install: Sequence[str],
    pytest_cmd: Sequence[str],
    python_cmd: Sequence[str],
    verbose: int = 0,
) -> list[ProbeResult]:
    """Probe versions in order with overlapping setup and execution stages.

    Uses two thread pools (setup and execution) coordinated via a queue. While workers
    in the execution pool run pytest tests, workers in the setup pool prepare the next
    set of environments. This allows I/O-bound setup (pip install) to overlap with
    CPU-light pytest execution.

    Preserves "first failure in order" semantics.

    Args:
        versions: Version strings to probe in order (assumed ordered, will process from start).
        project_root: Root directory of the project to test.
        dist_name: Distribution name to probe.
        max_workers: Maximum concurrent execution workers (setup workers auto-scaled).
        stop_on_first_fail: If True, stop probing on the first failure.
        keep_failed_venv: If True, keep venv directories for failed probes.
        extra_install: Additional packages to install.
        pytest_cmd: Pytest command to run.
        python_cmd: Python command for venv creation.
        verbose: Verbosity level 0-3. 0=quiet, 1=normal, 2+=worker details with timings.

    Returns:
        List of ProbeResult objects in order.

    Example:
        results = _pipelined_parallel_probe(
            versions=["8.0.0", "7.4.2"],
            project_root=Path("."),
            dist_name="pytest",
            max_workers=4,
            stop_on_first_fail=True,
            keep_failed_venv=False,
            extra_install=[],
            pytest_cmd=["-m", "pytest"],
            python_cmd=["python3.10"],
            verbose=2

        )
    """
    # Queue holds prepared environments ready for execution
    prepared_queue: queue.Queue[Optional[PreparedEnvironment]] = queue.Queue(
        maxsize=max_workers * 2
    )

    results: list[Optional[ProbeResult]] = [None] * len(versions)
    stop_event: threading.Event = threading.Event()
    next_setup_index: int = 0
    next_exec_index: int = 0
    worker_lock: threading.Lock = threading.Lock()

    def setup_worker(worker_id: int) -> None:
        """Continuously prepare environments from the versions list."""
        nonlocal next_setup_index
        while next_setup_index < len(versions) and not stop_event.is_set():
            with worker_lock:
                idx: int = next_setup_index
                next_setup_index += 1

            version: str = versions[idx]

            if verbose >= 2:
                print(
                    f"[Setup Worker {worker_id}] Starting setup for version {version}"
                )

            setup_start: float = time.time()
            prepared: PreparedEnvironment = _setup_pytest_environment(
                project_root=project_root,
                version=version,
                dist_name=dist_name,
                extra_install=extra_install,
                python_cmd=python_cmd,
            )
            setup_duration: float = time.time() - setup_start

            if verbose >= 2:
                status: str = "SUCCESS" if prepared.setup_passed else "FAILED"
                print(
                    f"[Setup Worker {worker_id}] Setup complete for {version} "
                    f"({setup_duration:.1f}s) - {status}"
                )

            prepared_queue.put(prepared)

    with cf.ThreadPoolExecutor(max_workers=max_workers) as exec_pool:
        with cf.ThreadPoolExecutor(max_workers=max(1, max_workers // 2)) as setup_pool:
            # Start setup workers
            # (pylint: disable=unused-variable)
            # The futures list is not explicitly awaited, but setup workers run
            # concurrently and coordinate with exec workers via the queue.
            _ = [
                setup_pool.submit(setup_worker, setup_worker_id)
                for setup_worker_id in range(max(1, max_workers // 2))
            ]

            exec_futures: dict[int, cf.Future[ProbeResult]] = {}

            def submit_execution(idx: int, exec_worker_id: int) -> None:
                """Submit an execution task for a prepared environment."""
                try:
                    prepared: PreparedEnvironment = prepared_queue.get(timeout=30)
                    if verbose >= 2:
                        print(
                            f"[Exec Worker {exec_worker_id}] Starting execution for version {prepared.version}"
                        )
                    exec_futures[idx] = exec_pool.submit(
                        _run_pytest_in_environment,
                        prepared_env=prepared,
                        project_root=project_root,
                        pytest_cmd=pytest_cmd,
                        keep_failed_venv=keep_failed_venv,
                        verbose=verbose,
                        worker_id=exec_worker_id,
                    )
                except queue.Empty:
                    pass

            # Fill initial execution queue
            exec_worker_counter: int = 0
            for idx in range(min(max_workers, len(versions))):
                submit_execution(idx, exec_worker_counter % max_workers)
                next_exec_index += 1
                exec_worker_counter += 1

            # Collect results in order
            collected_count: int = 0
            while collected_count < len(versions):
                if collected_count in exec_futures:
                    fut: cf.Future[ProbeResult] = exec_futures[collected_count]
                    res: ProbeResult = fut.result()
                    results[collected_count] = res
                    del exec_futures[collected_count]

                    if res.passed:
                        print(f"PASS: {res.version} ({res.duration_s:.1f} s)")
                    else:
                        print(
                            f"FAIL: {res.version} ({res.duration_s:.1f} s), rc={res.returncode}"
                        )
                        if res.stdout.strip():
                            print("  stdout:")
                            print(res.stdout.rstrip())
                        if res.stderr.strip():
                            print("  stderr:")
                            print(res.stderr.rstrip())
                        if keep_failed_venv:
                            print(f"  kept venv: {res.venv_dir}")

                        if stop_on_first_fail:
                            stop_event.set()
                            for f in exec_futures.values():
                                f.cancel()
                            break

                    # Submit next execution task if available
                    if next_exec_index < len(versions):
                        submit_execution(next_exec_index)
                        next_exec_index += 1
                else:
                    # Execution hasn't started yet, wait briefly
                    time.sleep(0.1)

                collected_count += 1

            # Signal setup workers to stop
            stop_event.set()

    out: list[ProbeResult] = []
    for r in results:
        if r is None:
            break
        out.append(r)
    return out


def _ordered_parallel_probe(
    *,
    versions: Sequence[str],
    project_root: Path,
    dist_name: str,
    max_workers: int,
    stop_on_first_fail: bool,
    keep_failed_venv: bool,
    extra_install: Sequence[str],
    pytest_cmd: Sequence[str],
    python_cmd: Sequence[str],
) -> list[ProbeResult]:
    """Probe versions in order, running up to max_workers concurrently.

    This is the legacy implementation that runs setup and execution sequentially per worker.
    For new code, use _pipelined_parallel_probe to enable overlapping setup and execution.

    This preserves "first failure in order" semantics, while still overlapping work.

    Args:
        versions: Version strings to probe in order.
        project_root: Root directory of the project to test.
        dist_name: Distribution name to probe.
        max_workers: Maximum concurrent workers.
        stop_on_first_fail: If True, stop on first failure.
        keep_failed_venv: If True, keep failed venv directories.
        extra_install: Additional packages to install.
        pytest_cmd: Pytest command to run.
        python_cmd: Python command for venv creation.

    Returns:
        List of ProbeResult objects in order.

    Example:
        results = _ordered_parallel_probe(
            versions=["8.0.0", "7.4.2"],
            project_root=Path("."),
            dist_name="pytest",
            max_workers=4,
            stop_on_first_fail=True,
            keep_failed_venv=False,
            extra_install=[],
            pytest_cmd=["-m", "pytest"],
            python_cmd=["python3.10"]
        )
    """
    results: list[Optional[ProbeResult]] = [None] * len(versions)
    next_submit: int = 0
    next_collect: int = 0

    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures: dict[int, cf.Future[ProbeResult]] = {}

        def submit(i: int) -> None:
            futures[i] = ex.submit(
                probe_pytest_version,
                project_root=project_root,
                pytest_version=versions[i],
                dist_name=dist_name,
                keep_failed_venv=keep_failed_venv,
                extra_install=extra_install,
                pytest_cmd=pytest_cmd,
                python_cmd=python_cmd,
            )

        while next_submit < len(versions) and len(futures) < max_workers:
            submit(next_submit)
            next_submit += 1

        while next_collect < len(versions):
            fut: cf.Future[ProbeResult] = futures[next_collect]
            res: ProbeResult = fut.result()
            results[next_collect] = res
            del futures[next_collect]

            if res.passed:
                print(f"PASS: {res.version} ({res.duration_s:.1f} s)")
            else:
                print(
                    f"FAIL: {res.version} ({res.duration_s:.1f} s), rc={res.returncode}"
                )
                if res.stdout.strip():
                    print("  stdout:")
                    print(res.stdout.rstrip())
                if res.stderr.strip():
                    print("  stderr:")
                    print(res.stderr.rstrip())
                if keep_failed_venv:
                    print(f"  kept venv: {res.venv_dir}")

            if stop_on_first_fail and (not res.passed):
                for f in futures.values():
                    f.cancel()
                break

            while next_submit < len(versions) and len(futures) < max_workers:
                submit(next_submit)
                next_submit += 1

            next_collect += 1

    out: list[ProbeResult] = []
    for r in results:
        if r is None:
            break
        out.append(r)
    return out


def main() -> int:
    """Entry point for the pytest compatibility probe."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory, default is current working directory.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        help="Python executable path or name on PATH for venv creation.",
    )
    parser.add_argument(
        "--python-version",
        type=str,
        default=None,
        help=(
            "Python version like 3.9 or 3.9.18. "
            "First tries pythonX.Y on PATH. "
            "If not found and pyenv is available, X.Y resolves to latest X.Y.Z, "
            "the version is installed if missing, then used for venv creation."
        ),
    )
    parser.add_argument(
        "--dist",
        type=str,
        required=True,
        help="Distribution name to probe (e.g., pytest, nbformat).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Max parallel venv probes to run concurrently.",
    )
    parser.add_argument(
        "--keep-failed-venv",
        action="store_true",
        help="Keep venv directories for failing probes to allow debugging.",
    )
    parser.add_argument(
        "--extra-install",
        nargs="*",
        default=[],
        help="Extra packages to install before running tests.",
    )
    parser.add_argument(
        "--pytest-args",
        nargs=argparse.REMAINDER,
        default=["-c", os.devnull],
        help="Arguments passed to pytest, default is `-c <devnull>`. All remaining arguments after this flag are passed to pytest.",
    )
    parser.add_argument(
        "--walk-major-then-refine",
        action="store_true",
        help=(
            "Walk major versions first (newest per major). "
            "When a major version fails, probe all patch versions to refine the boundary."
        ),
    )
    parser.add_argument(
        "--stop-on-first-fail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop probing on the first failure.",
    )
    parser.add_argument(
        "--start-version",
        type=str,
        default=None,
        help=(
            "Version to start probing from. Defaults to the installed version, "
            "falling back to the newest available."
        ),
    )
    parser.add_argument(
        "--exclude-versions",
        nargs="*",
        default=[],
        help="Specific versions to exclude from probing (e.g., 7.1.0 7.1.1).",
    )
    args: argparse.Namespace = parser.parse_args()

    if args.python and args.python_version:
        parser.error("Use only one of --python or --python-version.")

    project_root: Path = args.project_root.resolve()
    dist_name: str = args.dist
    max_workers: int = int(args.max_workers)
    keep_failed_venv: bool = bool(args.keep_failed_venv)
    extra_install: list[str] = list(args.extra_install)
    pytest_cmd: list[str] = ["-m", "pytest", *list(args.pytest_args)]
    stop_on_first_fail: bool = bool(args.stop_on_first_fail)
    walk_major_then_refine: bool = bool(args.walk_major_then_refine)
    python_cmd: list[str] = _resolve_python_cmd(args.python, args.python_version)

    all_versions: list[str] = _available_versions_via_pip_index(dist_name)
    exclude_versions: list[str] = list(args.exclude_versions or [])
    # Support both comma-separated and space-separated exclusions
    if exclude_versions:
        # If a single item contains commas, split by commas
        if len(exclude_versions) == 1 and "," in exclude_versions[0]:
            exclude_versions = [v.strip() for v in exclude_versions[0].split(",")]
        all_versions = _filter_excluded_versions(all_versions, exclude_versions)

    versions: list[str] = all_versions
    if walk_major_then_refine:
        versions = _latest_per_major(all_versions)
    installed: Optional[str] = _current_installed_version(dist_name)

    if args.start_version is not None:
        current_version = args.start_version
    elif installed is None:
        current_version = versions[0]
    else:
        current_version = installed

    current_index: int = 0
    for i, v in enumerate(versions):
        if v == current_version:
            current_index = i
            break
    else:
        print(
            f"Requested start version {current_version!r} not found in available "
            f"versions for {dist_name}."
        )
        return 2

    print(f"Current {dist_name} version (this interpreter): {installed or 'none'}")
    print(f"Starting probe at: {current_version}")
    print(f"Total available versions: {len(versions)}")
    if exclude_versions:
        print(f"Excluded versions: {', '.join(exclude_versions)}")
    print(f"Parallel workers: {max_workers}")
    print(f"Python command for venvs: {' '.join(python_cmd)}")
    print(f"Versions to test: {versions}")

    print(f"\nWalking down from {current_version} to older versions.")
    down_versions: list[str] = versions[current_index:]
    down_results: list[ProbeResult] = _pipelined_parallel_probe(
        versions=down_versions,
        project_root=project_root,
        dist_name=dist_name,
        max_workers=max_workers,
        stop_on_first_fail=stop_on_first_fail,
        keep_failed_venv=keep_failed_venv,
        extra_install=extra_install,
        pytest_cmd=pytest_cmd,
        python_cmd=python_cmd,
    )

    last_ok: Optional[str] = None
    down_failed: bool = False
    failed_version: Optional[str] = None
    for r in down_results:
        if r.passed:
            last_ok = r.version
        else:
            down_failed = True
            failed_version = r.version
            break

    # If walk_major_then_refine is enabled and a major version failed, refine at patch level
    if walk_major_then_refine and down_failed and failed_version is not None:
        match: Optional[re.Match[str]] = re.match(
            r"^(\d+)\.(\d+)\.(\d+)", failed_version
        )
        if match is not None:
            failed_major: str = match.group(1)
            print(
                f"\nMajor version {failed_major} failed at {failed_version}. "
                "Refining patch boundary..."
            )
            patch_versions: list[str] = _versions_for_major(versions, failed_major)
            if patch_versions:
                patch_results: list[ProbeResult] = _pipelined_parallel_probe(
                    versions=patch_versions,
                    project_root=project_root,
                    dist_name=dist_name,
                    max_workers=max_workers,
                    stop_on_first_fail=stop_on_first_fail,
                    keep_failed_venv=keep_failed_venv,
                    extra_install=extra_install,
                    pytest_cmd=pytest_cmd,
                    python_cmd=python_cmd,
                )
                for r in patch_results:
                    if r.passed:
                        last_ok = r.version
                    else:
                        break

    if down_failed and current_index > 0:
        print(f"\nWalking up from {current_version} to newer versions.")
        up_versions: list[str] = list(reversed(versions[:current_index]))
        _ = _pipelined_parallel_probe(
            versions=up_versions,
            project_root=project_root,
            dist_name=dist_name,
            max_workers=max_workers,
            stop_on_first_fail=stop_on_first_fail,
            keep_failed_venv=keep_failed_venv,
            extra_install=extra_install,
            pytest_cmd=pytest_cmd,
            python_cmd=python_cmd,
        )

    print(f"\nLast passing version on downward walk: {last_ok or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
