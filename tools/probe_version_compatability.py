#!/usr/bin/env python3
"""Probe pytest compatibility across multiple version selections.
TODO:
- when walking major version once found a fail go to patches and try to resolve the boundary

"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
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


def probe_pytest_version(
    *,
    project_root: Path,
    pytest_version: str,
    keep_failed_venv: bool,
    extra_install: Sequence[str],
    pytest_cmd: Sequence[str],
    python_cmd: Sequence[str],
) -> ProbeResult:
    """Create an isolated venv, install dependencies, run pytest, then clean up.

    Args:
        project_root: Root directory of the project to test, used as cwd for installs and pytest.
        pytest_version: Exact pytest version string to install, for example "8.4.2".
        keep_failed_venv: If True, keep the venv directory when the probe fails.
        extra_install: Additional packages to install after pytest and its immediate deps.
        pytest_cmd: Pytest command to run, excluding the python executable prefix.

    Returns:
        ProbeResult containing pass/fail, outputs, and timing.
    """
    venv_dir_str: str = tempfile.mkdtemp(prefix="pytest-venv.")
    venv_dir: Path = Path(venv_dir_str)

    t0: float = time.time()
    env: dict[str, str] = _base_env_for_venv(venv_dir)
    vpy: Path = _venv_python(venv_dir)

    stdout_all: list[str] = []
    stderr_all: list[str] = []
    install_stdout: list[str] = []
    install_stderr: list[str] = []
    passed: bool = False
    returncode: int = 1

    try:
        _make_venv(venv_dir, python_cmd)

        proc: subprocess.CompletedProcess[str]

        proc = _run(
            [str(vpy), "-m", "pip", "install", "-U", "pip"],
            cwd=project_root,
            env=env,
            check=False,
        )
        install_stdout.append(proc.stdout)
        install_stderr.append(proc.stderr)
        if proc.returncode != 0:
            returncode = proc.returncode
            passed = False
            return ProbeResult(
                version=pytest_version,
                passed=passed,
                returncode=returncode,
                duration_s=time.time() - t0,
                venv_dir=venv_dir,
                stdout="",
                stderr="".join(install_stderr),
            )

        proc = _run(
            [
                str(vpy),
                "-m",
                "pip",
                "install",
                "-U",
                f"pytest=={pytest_version}",
                "pytest-timeout",
                "nbformat",
            ],
            cwd=project_root,
            env=env,
            check=False,
        )
        install_stdout.append(proc.stdout)
        install_stderr.append(proc.stderr)
        if proc.returncode != 0:
            returncode = proc.returncode
            passed = False
            return ProbeResult(
                version=pytest_version,
                passed=passed,
                returncode=returncode,
                duration_s=time.time() - t0,
                venv_dir=venv_dir,
                stdout="",
                stderr="".join(install_stderr),
            )

        if extra_install:
            proc = _run(
                [str(vpy), "-m", "pip", "install", *list(extra_install)],
                cwd=project_root,
                env=env,
                check=False,
            )
            install_stdout.append(proc.stdout)
            install_stderr.append(proc.stderr)
            if proc.returncode != 0:
                returncode = proc.returncode
                passed = False
                return ProbeResult(
                    version=pytest_version,
                    passed=passed,
                    returncode=returncode,
                    duration_s=time.time() - t0,
                    venv_dir=venv_dir,
                    stdout="",
                    stderr="".join(install_stderr),
                )

        proc = _run(
            [str(vpy), "-m", "pip", "install", "-e", "."],
            cwd=project_root,
            env=env,
            check=False,
        )
        install_stdout.append(proc.stdout)
        install_stderr.append(proc.stderr)
        if proc.returncode != 0:
            returncode = proc.returncode
            passed = False
            return ProbeResult(
                version=pytest_version,
                passed=passed,
                returncode=returncode,
                duration_s=time.time() - t0,
                venv_dir=venv_dir,
                stdout="",
                stderr="".join(install_stderr),
            )

        cache_dir: str = f"/tmp/pytest_cache/{pytest_version}"
        proc = _run(
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
            version=pytest_version,
            passed=passed,
            returncode=returncode,
            duration_s=time.time() - t0,
            venv_dir=venv_dir,
            stdout="".join(stdout_all),
            stderr="".join(stderr_all) + "".join(install_stderr),
        )
    finally:
        if passed or not keep_failed_venv:
            shutil.rmtree(venv_dir, ignore_errors=True)


def _ordered_parallel_probe(
    *,
    versions: Sequence[str],
    project_root: Path,
    max_workers: int,
    stop_on_first_fail: bool,
    keep_failed_venv: bool,
    extra_install: Sequence[str],
    pytest_cmd: Sequence[str],
    python_cmd: Sequence[str],
) -> list[ProbeResult]:
    """Probe versions in order, running up to max_workers concurrently.

    This preserves "first failure in order" semantics, while still overlapping work.
    """
    import concurrent.futures as cf

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
        default="pytest",
        help="Distribution name to probe, default is pytest.",
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
        default=["matplotlib"],
        help="Extra packages to install before running tests.",
    )
    parser.add_argument(
        "--pytest-args",
        nargs="*",
        default=["-c", os.devnull],
        help="Arguments passed to pytest, default is `-c <devnull>`.",
    )
    parser.add_argument(
        "--major-latest-only",
        action="store_true",
        help="Probe only the newest version for each major series.",
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
    python_cmd: list[str] = _resolve_python_cmd(args.python, args.python_version)

    versions: list[str] = _available_versions_via_pip_index(dist_name)
    if args.major_latest_only:
        versions = _latest_per_major(versions)
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
    print(f"Parallel workers: {max_workers}")
    print(f"Python command for venvs: {' '.join(python_cmd)}")

    print(f"\nWalking down from {current_version} to older versions.")
    down_versions: list[str] = versions[current_index:]
    down_results: list[ProbeResult] = _ordered_parallel_probe(
        versions=down_versions,
        project_root=project_root,
        max_workers=max_workers,
        stop_on_first_fail=stop_on_first_fail,
        keep_failed_venv=keep_failed_venv,
        extra_install=extra_install,
        pytest_cmd=pytest_cmd,
        python_cmd=python_cmd,
    )

    last_ok: Optional[str] = None
    down_failed: bool = False
    for r in down_results:
        if r.passed:
            last_ok = r.version
        else:
            down_failed = True
            break

    if down_failed and current_index > 0:
        print(f"\nWalking up from {current_version} to newer versions.")
        up_versions: list[str] = list(reversed(versions[:current_index]))
        _ = _ordered_parallel_probe(
            versions=up_versions,
            project_root=project_root,
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
