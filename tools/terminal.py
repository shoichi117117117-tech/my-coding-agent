import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path
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
    """Safe execution wrapper for terminal commands with on-demand venv support."""

    def __init__(self, default_timeout: int = 30):
        """
        Initialize the TerminalRunner.

        :param default_timeout: Maximum time in seconds allowed for command execution (default: 30s).
        """
        self.default_timeout = default_timeout

    def _ensure_venv(self, cwd_path: Path) -> Path:
        """
        Ensure a Python virtual environment (.venv) exists in the workspace.
        Create it dynamically only when needed.

        :param cwd_path: Workspace directory path.
        :return: Path to the python executable inside .venv.
        """
        venv_dir = cwd_path / ".venv"

        # Create .venv if it does not exist yet
        if not venv_dir.exists():
            subprocess.run(
                [sys.executable, "-m", "venv", str(venv_dir)],
                check=True,
                capture_output=True,
            )

        # Return platform-specific python binary path inside .venv
        if sys.platform == "win32":
            return venv_dir / "Scripts" / "python.exe"
        return venv_dir / "bin" / "python"

    def run(
        self, command: str, timeout: Optional[int] = None, cwd: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute a terminal command safely and capture its execution status and outputs.
        If command contains 'python' or 'pytest', it automatically routes execution
        through a local .venv (lazy initialization).

        :param command: CLI command string to run (e.g., "pytest", "python script.py").
        :param timeout: Optional override for default timeout in seconds.
        :param cwd: Optional working directory path where the command should execute.
        :return: ExecutionResult dataclass containing status, outputs, and exit code.
        """
        active_timeout = timeout if timeout is not None else self.default_timeout
        cwd_path = Path(cwd) if cwd else Path.cwd()

        # Intercept Python execution to utilize dynamic local .venv
        cmd_lower = command.lower()
        if "python" in cmd_lower or "pytest" in cmd_lower:
            try:
                venv_python = self._ensure_venv(cwd_path)

                # Replace generic 'python' or 'pytest' calls with local venv executable
                if command.startswith("python "):
                    command = command.replace("python", f'"{venv_python}"', 1)
                elif command.startswith("python -m"):
                    command = command.replace("python", f'"{venv_python}"', 1)
                elif command.startswith("pytest"):
                    command = f'"{venv_python}" -m pytest' + command[6:]
            except Exception as venv_err:
                return ExecutionResult(
                    success=False,
                    returncode=-1,
                    stdout="",
                    stderr=f"❌ Failed to setup virtual environment: {str(venv_err)}",
                    command=command,
                )

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
    print("⚡ Testing TerminalRunner module with dynamic venv...")
    runner = TerminalRunner(default_timeout=5)

    # Test 1: Successful command (Python call triggers lazy .venv setup)
    res_success = runner.run("python --version")
    print(
        f"Test 1 (Success Check): success={res_success.success}, command={res_success.command}, stdout={res_success.stdout.strip()}"
    )

    # Test 2: Non-python command (Bypasses venv setup entirely)
    res_echo = runner.run("echo Hello World")
    print(
        f"Test 2 (Non-Python Check): success={res_echo.success}, stdout={res_echo.stdout.strip()}"
    )