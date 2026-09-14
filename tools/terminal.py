import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExecutionResult:
    """Data structure holding the outcome of a command execution."""

    success: bool
    returncode: int
    stdout: str
    stderr: str
    command: str


class TerminalRunner:
    """Safe execution wrapper for terminal commands (e.g., tests, linter, build scripts)."""

    def __init__(self, default_timeout: int = 30):
        """
        Initialize the TerminalRunner.

        :param default_timeout: Maximum time in seconds allowed for command execution (default: 30s).
        """
        self.default_timeout = default_timeout

    def run(
        self, command: str, timeout: Optional[int] = None, cwd: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute a terminal command safely and capture its execution status and outputs.

        :param command: CLI command string to run (e.g., "pytest", "python -m unittest").
        :param timeout: Optional override for default timeout in seconds.
        :param cwd: Optional working directory path where the command should execute.
        :return: ExecutionResult dataclass containing status, outputs, and exit code.
        """
        active_timeout = timeout if timeout is not None else self.default_timeout

        try:
            # Execute command using subprocess
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=active_timeout,
                cwd=cwd,
            )

            is_success = result.returncode == 0
            return ExecutionResult(
                success=is_success,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                command=command,
            )

        except subprocess.TimeoutExpired as e:
            error_msg = (
                f"⏱️ Command timed out after {active_timeout} seconds: {command}"
            )
            return ExecutionResult(
                success=False,
                returncode=-1,
                stdout=e.stdout or "",
                stderr=error_msg,
                command=command,
            )

        except Exception as e:
            error_msg = f"❌ Execution failed with exception: {str(e)}"
            return ExecutionResult(
                success=False,
                returncode=-1,
                stdout="",
                stderr=error_msg,
                command=command,
            )


# --- Quick Standalone Test ---
if __name__ == "__main__":
    print("⚡ Testing TerminalRunner module...")
    runner = TerminalRunner(default_timeout=5)

    # Test 1: Successful command
    res_success = runner.run("python --version")
    print(
        f"Test 1 (Success Check): success={res_success.success}, stdout={res_success.stdout.strip()}"
    )

    # Test 2: Failing command
    res_fail = runner.run("python non_existent_script.py")
    print(
        f"Test 2 (Failure Check): success={res_fail.success}, stderr_snippet={res_fail.stderr[:50].strip()}"
    )