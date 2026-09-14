import os
import re
from pathlib import Path
from typing import Optional, Dict


class CodeEditor:
    """Utility class for scanning project files and extracting/updating code."""

    def __init__(self, workspace_root: Optional[str] = None):
        """
        Initialize CodeEditor with a target workspace root directory.

        :param workspace_root: Root path of the target code directory (default: current working directory).
        """
        self.workspace_root = Path(workspace_root or os.getcwd()).resolve()

    def read_file(self, relative_path: str) -> str:
        """
        Read content from a file inside the workspace.

        :param relative_path: Path relative to workspace root.
        :return: String content of the file or empty string if file doesn't exist.
        """
        file_path = self.workspace_root / relative_path
        if not file_path.exists():
            return ""

        try:
            return file_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"⚠️ Error reading file {relative_path}: {e}")
            return ""

    def write_file(self, relative_path: str, content: str) -> bool:
        """
        Write or update content to a file inside the workspace safely.

        :param relative_path: Path relative to workspace root.
        :param content: Text/code content to write.
        :return: True if successful, False otherwise.
        """
        file_path = self.workspace_root / relative_path
        try:
            # Ensure parent directories exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            print(f"💾 File updated: `{relative_path}`")
            return True
        except Exception as e:
            print(f"❌ Error writing file {relative_path}: {e}")
            return False

    def scan_workspace(self, extensions: Optional[list] = None, max_files: int = 20) -> Dict[str, str]:
        """
        Scan workspace files to provide multi-file context to Gemini.

        :param extensions: List of file extensions to include (e.g., ['.py', '.md']).
        :param max_files: Maximum number of files to read.
        :return: Dictionary mapping relative file path to its content snippet.
        """
        target_exts = extensions or [".py", ".json", ".md", ".txt"]
        scanned_files = {}

        count = 0
        for root, dirs, files in os.walk(self.workspace_root):
            # Skip hidden folders, venv, git, node_modules
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ["venv", "__pycache__", "node_modules"]]

            for file in files:
                if count >= max_files:
                    break
                ext = Path(file).suffix
                if ext in target_exts:
                    rel_path = str(Path(root, file).relative_to(self.workspace_root))
                    scanned_files[rel_path] = self.read_file(rel_path)
                    count += 1

        return scanned_files

    @staticmethod
    def extract_code_block(llm_response: str) -> str:
        """
        Extract code from markdown code fences inside LLM responses.

        :param llm_response: Full text response returned by Gemini.
        :return: Extracted raw code string.
        """
        # Match pattern ```python ... ``` or generic ``` ... ```
        pattern = r"```(?:\w+)?\n(.*?)```"
        matches = re.findall(pattern, llm_response, re.DOTALL)

        if matches:
            # Return the largest code block if multiple exist
            return max(matches, key=len).strip()

        # Fallback: if no code fence found, return raw response cleaned up
        return llm_response.strip()


# --- Quick Standalone Test ---
if __name__ == "__main__":
    print("🛠️ Testing CodeEditor module...")
    editor = CodeEditor()

    # Test code extraction helper
    sample_llm_text = "Here is the code:\n```python\ndef hello():\n    print('Hello World')\n```"
    extracted = editor.extract_code_block(sample_llm_text)
    print("Extracted Code:\n" + extracted)