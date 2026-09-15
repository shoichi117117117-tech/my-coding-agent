import os
from pathlib import Path
from typing import Optional, Dict, List


class CodeEditor:
    """Workspace manager and file system interface for Gemini Agent tools."""

    def __init__(self, workspace_root: Optional[str] = None):
        """
        Initialize CodeEditor with a target workspace root directory path.

        :param workspace_root: Absolute or relative path to the target project directory.
        """
        if workspace_root:
            target_path = Path(workspace_root).expanduser().resolve()
        else:
            target_path = (Path.home() / "projects_with_agent").resolve()

        self.workspace_root = target_path
        # Ensure workspace root exists
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        print(f"📁 [Workspace Initialized] -> `{self.workspace_root}`")

    def _resolve_path(self, relative_or_abs_path: str) -> Path:
        """
        Resolve input string into an absolute Path inside the workspace root.

        :param relative_or_abs_path: Relative file path or explicit absolute path.
        :return: Resolved absolute Path object.
        """
        path = Path(relative_or_abs_path.strip())
        return (self.workspace_root / path) if not path.is_absolute() else path.resolve()

    def write_file(self, file_path: str, content: str) -> str:
        """
        Create or update a file in the workspace with given content.
        Automatically creates missing parent directories.

        :param file_path: Relative file path within workspace (e.g., 'src/utils/math.py').
        :param content: Full text/code content to write into the file.
        :return: Status message confirming success or failure.
        """
        if not str(file_path).strip():
            return "Error: File path cannot be empty."

        target_path = self._resolve_path(file_path)

        try:
            # Automatically create parent directory trees if they don't exist
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content, encoding="utf-8")
            print(f"💾 [Tool Call: write_file] Success -> `{target_path}`")
            return f"Successfully written to '{file_path}'"
        except Exception as e:
            error_msg = f"Failed to write file '{file_path}': {str(e)}"
            print(f"❌ [Tool Call: write_file] {error_msg}")
            return error_msg

    def read_file(self, file_path: str) -> str:
        """
        Read text content of a specific file in the workspace.

        :param file_path: Relative path to the file to read.
        :return: Text content of the file or error description.
        """
        target_path = self._resolve_path(file_path)
        if not target_path.exists():
            return f"Error: File '{file_path}' does not exist."
        if target_path.is_dir():
            return f"Error: Path '{file_path}' is a directory, not a file."

        try:
            return target_path.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading file '{file_path}': {str(e)}"

    def make_directory(self, dir_path: str) -> str:
        """
        Create a directory or subdirectories inside the workspace.

        :param dir_path: Relative directory path (e.g., 'src/components').
        :return: Status message.
        """
        target_path = self._resolve_path(dir_path)
        try:
            target_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 [Tool Call: make_directory] Created -> `{target_path}`")
            return f"Directory created: '{dir_path}'"
        except Exception as e:
            return f"Failed to create directory '{dir_path}': {str(e)}"

    def list_files(self, relative_dir: str = ".") -> str:
        """
        List all files and folders inside a specific workspace directory.

        :param relative_dir: Target subdirectory relative to workspace root. Default is '.'.
        :return: Multiline string showing file structure tree.
        """
        target_dir = self._resolve_path(relative_dir)
        if not target_dir.exists() or not target_dir.is_dir():
            return f"Error: Directory '{relative_dir}' does not exist."

        file_list = []
        for root, dirs, files in os.walk(target_dir):
            # Skip hidden and dependency folders
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ["venv", "__pycache__", "node_modules"]]
            rel_root = Path(root).relative_to(self.workspace_root)
            for f in files:
                file_list.append(str(rel_root / f))

        if not file_list:
            return f"No files found in '{relative_dir}'."

        return "\n".join(file_list)


# --- Standalone Verification Test ---
if __name__ == "__main__":
    print("🛠️ Testing Function Calling CodeEditor...")
    test_editor = CodeEditor(workspace_root="./test_workspace")

    # Direct function invocation test
    print(test_editor.make_directory("src/core"))
    print(test_editor.write_file("src/core/calculator.py", "def multiply(a, b):\n    return a * b\n"))
    print(test_editor.read_file("src/core/calculator.py"))
    print("\nWorkspace Files:")
    print(test_editor.list_files())