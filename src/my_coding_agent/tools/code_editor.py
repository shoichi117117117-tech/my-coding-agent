"""Workspace manager and file system interface for multi-agent system tools."""

import os
from pathlib import Path
from typing import List, Optional


class CodeEditor:
    """Workspace manager providing isolated file system operations for AI agents."""

    def __init__(self, workspace_root: Optional[str] = None):
        if workspace_root:
            target_path = Path(workspace_root).expanduser().resolve()
        else:
            target_path = Path.cwd().resolve()

        self.workspace_root: Path = target_path
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, relative_or_abs_path: str) -> Path:
        """Resolve path and ensure it remains strictly within workspace boundaries."""
        clean_path_str = str(relative_or_abs_path).strip()
        path = Path(clean_path_str)

        if path.is_absolute():
            resolved = path.resolve()
        else:
            resolved = (self.workspace_root / path).resolve()

        # Security check: Prevent path traversal outside workspace
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError:
            # Fallback to file name isolated inside workspace root if traversal attempted
            return self.workspace_root / path.name

        return resolved

    def write_file(self, file_path: str, content: str) -> str:
        """Write or overwrite text content to a file safely within workspace."""
        if not str(file_path).strip():
            return "Error: File path cannot be empty."

        target_path = self._resolve_path(file_path)
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content, encoding="utf-8")
            return f"Successfully written to '{target_path.relative_to(self.workspace_root)}'"
        except Exception as e:
            return f"Failed to write file '{file_path}': {str(e)}"

    def read_file(self, file_path: str) -> str:
        """Read and return utf-8 text content from a file."""
        if not str(file_path).strip():
            return "Error: File path cannot be empty."

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
        """Create target directory hierarchy safely within workspace."""
        if not str(dir_path).strip():
            return "Error: Directory path cannot be empty."

        target_path = self._resolve_path(dir_path)
        try:
            target_path.mkdir(parents=True, exist_ok=True)
            return f"Directory created: '{target_path.relative_to(self.workspace_root)}'"
        except Exception as e:
            return f"Failed to create directory '{dir_path}': {str(e)}"

    def list_files(self, relative_dir: str = ".") -> str:
        """List relative paths of all files in directory, ignoring hidden/temp folders."""
        target_dir = self._resolve_path(relative_dir)

        if not target_dir.exists() or not target_dir.is_dir():
            return f"Error: Directory '{relative_dir}' does not exist."

        ignored_dirs = {
            "venv",
            ".venv",
            "__pycache__",
            "node_modules",
            ".git",
            ".pytest_cache",
            ".idea",
            ".vscode",
        }

        file_list: List[str] = []
        for root, dirs, files in os.walk(target_dir):
            # Prune ignored directories in-place
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ignored_dirs]

            try:
                rel_root = Path(root).relative_to(self.workspace_root)
            except ValueError:
                rel_root = Path(root)

            for file in files:
                if not file.startswith("."):
                    file_list.append(str(rel_root / file))

        if not file_list:
            return f"No files found in '{relative_dir}'."

        return "\n".join(file_list)

    def search_files(self, keyword: str) -> str:
        """Search workspace files matching pattern or keyword."""
        if not keyword.strip():
            return "Error: Keyword cannot be empty."

        all_files_output = self.list_files()
        if all_files_output.startswith("Error") or all_files_output.startswith("No files"):
            return all_files_output

        matches = [f for f in all_files_output.split("\n") if keyword.lower() in f.lower()]
        if not matches:
            return f"No files matching keyword '{keyword}' found."

        return "\n".join(matches)

    def delete_file(self, file_path: str) -> str:
        """Safely delete a file within the workspace root."""
        if not str(file_path).strip():
            return "Error: File path cannot be empty."

        target_path = self._resolve_path(file_path)

        if not target_path.exists():
            return f"Error: File '{file_path}' does not exist."

        if target_path.is_dir():
            return f"Error: Path '{file_path}' is a directory. Use directory tool to handle folders."

        try:
            target_path.unlink()
            return f"Successfully deleted file: '{target_path.relative_to(self.workspace_root)}'"
        except Exception as e:
            return f"Error deleting file '{file_path}': {str(e)}"


# =====================================================================
# Shared Singleton Instance & Exported Public Functions for Agent Tools
# =====================================================================
_default_editor = CodeEditor()


def write_file(file_path: str, content: str) -> str:
    """Public tool wrapper to write text content into a workspace file."""
    return _default_editor.write_file(file_path, content)


def read_file(file_path: str) -> str:
    """Public tool wrapper to read content from a workspace file."""
    return _default_editor.read_file(file_path)


def make_directory(dir_path: str) -> str:
    """Public tool wrapper to create a directory within workspace."""
    return _default_editor.make_directory(dir_path)


def list_files(relative_dir: str = ".") -> str:
    """Public tool wrapper to list workspace files."""
    return _default_editor.list_files(relative_dir)


def search_files(keyword: str) -> str:
    """Public tool wrapper to search workspace files by keyword or extension."""
    return _default_editor.search_files(keyword)

def delete_file(file_path: str) -> str:
    """public tool wrapper to delete unneccesary files"""
    return _default_editor.delete_file(file_path)


# =====================================================================
# Standalone Quick Test
# =====================================================================
if __name__ == "__main__":
    print("--- Running Quick Test for tools/code_editor.py ---")
    test_dir = "test_workspace_dir"
    test_file = f"{test_dir}/hello.py"

    print(make_directory(test_dir))
    print(write_file(test_file, "# Hello World\nprint('Hello World')\n"))
    print("Read File Content:\n" + read_file(test_file))
    print("Search Result:\n" + search_files("hello"))