"""Demo harness for the pytest-notebook-test plugin."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


@dataclass(frozen=True, slots=True)
class DemoNotebook:
    """Notebook demo target description.

    Args:
        name: Short label used in the demo report.
        notebook: Path to the source notebook on disk.
        cli_args: Additional pytest CLI arguments to use for this demo.

    Example:
        demo = DemoNotebook(
            name="simple",
            notebook=Path("tests/notebooks/test_simple.ipynb"),
            cli_args=(),
        )
    """

    name: str
    """Short label used in the demo report."""

    notebook: Path
    """Path to the source notebook on disk."""

    cli_args: tuple[str, ...] = ()
    """Additional pytest CLI arguments to use for this demo."""


def run_demo() -> int:
    """Run demo notebooks through pytest and print a short report.

    Returns:
        Exit status code (0 for success, 1 for any failures).

    Example:
        python run_demo.py
    """
    repo_root = Path(__file__).resolve().parent
    notebooks_dir = repo_root / "tests" / "notebooks"
    demos = [
        DemoNotebook(
            name="simple",
            notebook=notebooks_dir / "test_simple.ipynb",
            cli_args=(),
        ),
        DemoNotebook(
            name="async-exec",
            notebook=notebooks_dir / "test_async_exec_mode.ipynb",
            cli_args=(),
        ),
        DemoNotebook(
            name="sync-exec",
            notebook=notebooks_dir / "test_sync_exec_mode.ipynb",
            cli_args=(
                "--notebook-exec-mode=sync",
                "--notebook-keep-generated={generated_dir}",
            ),
        ),
    ]
    exit_code = 0
    for demo in demos:
        print(f"\n=== Demo: {demo.name} ===")
        if not demo.notebook.exists():
            raise FileNotFoundError(f"Missing demo notebook: {demo.notebook}")
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            target_notebook = tmp_path / demo.notebook.name
            shutil.copy2(demo.notebook, target_notebook)
            generated_dir: Path | None = None
            resolved_args: list[str] = []
            for arg in demo.cli_args:
                if "{generated_dir}" in arg:
                    if generated_dir is None:
                        generated_dir = tmp_path / "generated"
                        generated_dir.mkdir(parents=True, exist_ok=True)
                    resolved_args.append(arg.replace("{generated_dir}", str(generated_dir)))
                else:
                    resolved_args.append(arg)
            cmd = [
                "pytest",
                "-q",
                "-p",
                "pytest_notebook_test.plugin",
                "--notebook-dir",
                str(tmp_path),
                str(tmp_path),
                *resolved_args,
            ]
            print(f"Notebook: {demo.notebook.name}")
            print(f"Command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                cwd=repo_root,
                env={
                    **os.environ,
                    "PYTHONPATH": f"{repo_root}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
                },
            )
            status = "PASS" if result.returncode == 0 else "FAIL"
            print(f"Result: {status}")
            if generated_dir is not None:
                print(f"Generated scripts: {generated_dir}")
            if result.returncode != 0:
                exit_code = 1
                print("---- pytest stdout ----")
                print(result.stdout.rstrip())
                print("---- pytest stderr ----")
                print(result.stderr.rstrip())
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run_demo())
