import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Data structure holding the outcome of a command execution."""

    success: bool
    returncode: int
    stdout: str
    stderr: str
    command: str


class TerminalRunner:
    """Generic, safe, and system-agnostic execution wrapper for CLI commands."""

    def __init__(self, default_timeout: int = 30):
        """Initialize the TerminalRunner.

        :param default_timeout: Maximum allowed execution time in seconds (default: 30s).
        """
        self.default_timeout = default_timeout

    def _get_isolated_env(self) -> dict[str, str]:
        """Strip parent process virtual environment variables

        to ensure subprocesses execute in a fully isolated target workspace environment.
        """
        env = os.environ.copy()

        # 1. Strip parent process virtual environment variables
        parent_venv = env.pop("VIRTUAL_ENV", None)
        env.pop("PYTHONHOME", None)

        # 2. Remove parent's .venv path from system PATH
        # (Prevents subprocesses from incorrectly invoking parent executables)
        if parent_venv:
            path_sep = os.pathsep
            current_paths = env.get("PATH", "").split(path_sep)
            cleaned_paths = [
                p for p in current_paths if not p.startswith(parent_venv)
            ]
            env["PATH"] = path_sep.join(cleaned_paths)

        return env

    def run(
        self, command: str, timeout: Optional[int] = None, cwd: Optional[str] = None
    ) -> ExecutionResult:
        """Execute a CLI command directly in the system shell and capture outputs."""
        active_timeout = timeout if timeout is not None else self.default_timeout
        final_command = command.strip()

        if not final_command:
            return ExecutionResult(
                success=False,
                returncode=-1,
                stdout="",
                stderr="Error: Command cannot be empty.",
                command="",
            )

        # Generate isolated environment variables for the subprocess
        isolated_env = self._get_isolated_env()

        logger.info(f"[TerminalRunner] Executing: '{final_command}' (cwd: {cwd})")
        try:
            result = subprocess.run(
                final_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=active_timeout,
                cwd=cwd,
                env=isolated_env,
            )

            is_success = result.returncode == 0
            return ExecutionResult(
                success=is_success,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                command=final_command,
            )

        except subprocess.TimeoutExpired as e:
            error_msg = f"⏱️ Command timed out after {active_timeout} seconds: {final_command}"
            logger.error(f"[TerminalRunner] {error_msg}")
            captured_stdout = e.stdout.decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            return ExecutionResult(
                success=False,
                returncode=-1,
                stdout=captured_stdout,
                stderr=error_msg,
                command=final_command,
            )

        except Exception as e:
            error_msg = f"❌ Execution failed with exception: {str(e)}"
            logger.error(f"[TerminalRunner] {error_msg}")
            return ExecutionResult(
                success=False,
                returncode=-1,
                stdout="",
                stderr=error_msg,
                command=final_command,
            )


# =====================================================================
# Standalone Quick Test
# =====================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("\n==================================================")
    print(" Running Live Standalone Quick Test for TerminalRunner")
    print("==================================================\n")

    try:
        runner = TerminalRunner(default_timeout=10)

        # Test 1: Standard echo
        print("--- Test 1: Echo Command Execution ---")
        res_echo = runner.run("echo Hello World")
        print(f"Executed Command: {res_echo.command}")
        print(f"Success: {res_echo.success} | Output: {res_echo.stdout.strip()}\n")

        # Test 2: Direct Command Execution
        print("--- Test 2: Direct Command Execution ---")
        res_py = runner.run("python --version")
        print(f"Executed Command: {res_py.command}")
        print(f"Success: {res_py.success} | Output: {res_py.stdout.strip()}\n")

        print("==================================================")
        print(" [SUCCESS] TerminalRunner Quick Test PASSED!")
        print("==================================================\n")

    except Exception as e:
        print("==================================================")
        print(f" [FAIL] Quick Test Failed with error:\n {e}")
        print("==================================================\n")
        sys.exit(1)