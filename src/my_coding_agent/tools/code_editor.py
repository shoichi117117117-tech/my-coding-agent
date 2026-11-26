"""Workspace manager and file system interface for multi-agent system tools."""

import ast
import json
import os
import re
import tomllib
from pathlib import Path
from typing import Any, List, Optional


def _normalize_dist_name(name: str) -> str:
    """Normalize a PEP 508 distribution name to its default Python package directory name."""
    return re.sub(r"[-_.]+", "_", name).strip("_").lower()


def _extract_readme_path(project_section: dict[str, Any]) -> Optional[str]:
    """Extract the configured readme file path from `[project].readme`, if any."""
    readme = project_section.get("readme")
    if isinstance(readme, str) and readme.strip():
        return readme.strip()
    if isinstance(readme, dict):
        file_val = readme.get("file")
        if isinstance(file_val, str) and file_val.strip():
            return file_val.strip()
    return None


def validate_uv_project(workspace_root: Path) -> List[str]:
    """Inspect `pyproject.toml` and workspace files for common `uv sync` / `hatchling` build traps.

    Returns a list of human-readable diagnostic issues (empty if valid or no `pyproject.toml`).
    """
    pyproject_path = workspace_root / "pyproject.toml"
    if not pyproject_path.exists():
        return []

    try:
        parsed = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except Exception as e:
        return [f"Invalid pyproject.toml syntax: {e}"]

    issues: List[str] = []
    project_section = parsed.get("project", {})
    if not isinstance(project_section, dict):
        project_section = {}

    if "dependency-groups" in project_section:
        issues.append(
            "Invalid placement: 'dependency-groups' is nested inside [project]. "
            "Move it to a top-level [dependency-groups] table."
        )

    dep_groups = parsed.get("dependency-groups")
    if dep_groups is not None:
        if not isinstance(dep_groups, dict) or any(
            not isinstance(v, list) or not all(isinstance(item, str) for item in v)
            for v in dep_groups.values()
        ):
            issues.append(
                "Invalid [dependency-groups] format: each group must be an array of PEP 508 strings (e.g. dev = [\"pytest>=8.0.0\"])."
            )

    readme_rel = _extract_readme_path(project_section)
    if readme_rel and not (workspace_root / readme_rel).is_file():
        issues.append(
            f"Missing readme file '{readme_rel}' declared in [project].readme; "
            f"hatchling will fail with OSError unless '{readme_rel}' is created or readme is removed."
        )

    tool_uv = parsed.get("tool", {}).get("uv", {}) if isinstance(parsed.get("tool"), dict) else {}
    uv_package_disabled = isinstance(tool_uv, dict) and tool_uv.get("package") is False

    build_system = parsed.get("build-system", {})
    build_backend = build_system.get("build-backend", "") if isinstance(build_system, dict) else ""

    if isinstance(build_backend, str) and build_backend.startswith("hatchling") and not uv_package_disabled:
        hatch_wheel = (
            parsed.get("tool", {})
            .get("hatch", {})
            .get("build", {})
            .get("targets", {})
            .get("wheel", {})
        )
        if not isinstance(hatch_wheel, dict):
            hatch_wheel = {}

        packages = hatch_wheel.get("packages")
        only_include = hatch_wheel.get("only-include")

        if isinstance(packages, list) and packages:
            missing_pkgs = [p for p in packages if not (workspace_root / str(p)).exists()]
            if missing_pkgs:
                issues.append(
                    f"Configured [tool.hatch.build.targets.wheel] packages {missing_pkgs} do not exist in workspace."
                )
        elif isinstance(only_include, list) and only_include:
            missing_inc = [p for p in only_include if not (workspace_root / str(p)).exists()]
            if missing_inc:
                issues.append(
                    f"Configured [tool.hatch.build.targets.wheel] only-include paths {missing_inc} do not exist in workspace."
                )
        else:
            dist_name = str(project_section.get("name", ""))
            norm_name = _normalize_dist_name(dist_name) if dist_name else ""
            has_pkg_dir = bool(
                norm_name
                and (
                    (workspace_root / norm_name).is_dir()
                    or (workspace_root / "src" / norm_name).is_dir()
                )
            )
            if not has_pkg_dir:
                issues.append(
                    f"Hatchling build-backend is enabled for project '{dist_name or 'unknown'}', "
                    f"but no matching package directory ('{norm_name}/' or 'src/{norm_name}/') or "
                    "[tool.hatch.build.targets.wheel] packages exists. Either create the package directory, "
                    "configure packages, or for flat scripts omit [build-system] and set [tool.uv] package = false."
                )

    return issues


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
        except ValueError as err:
            raise PermissionError(
                f"Access denied: path '{relative_or_abs_path}' resolves outside workspace root."
            ) from err

        return resolved

    def write_file(self, file_path: str, content: str) -> str:
        """Write file with automated normalized pre-flight syntax validation."""
        if not str(file_path).strip():
            return "Error: File path cannot be empty."

        try:
            target_path = self._resolve_path(file_path)
        except PermissionError as e:
            return f"Error: {e}"
        ext = target_path.suffix.lower()

        # --- NORMALIZED PRE-FLIGHT SYNTAX VALIDATION GATE ---
        validation_error = None

        try:
            if ext == ".py":
                # Check Python syntax without executing the code
                ast.parse(content)

            elif ext == ".toml":
                # 1. Catch the forbidden table header via regex (ignores spaces/formatting tricks)
                if re.search(r"\[\s*project\.dependencies\s*\]", content, re.IGNORECASE):
                    raise ValueError(
                        "Forbidden [project.dependencies] table header detected. "
                        "Dependencies must be specified as an array inside the [project] section, e.g.:\n"
                        "dependencies = [\n    \"pkg>=1.0\"\n]"
                    )

                # 2. Standard TOML parser check & structural array verification
                parsed_toml = tomllib.loads(content)
                project_section = parsed_toml.get("project", {})
                if isinstance(project_section, dict):
                    if "dependencies" in project_section and not isinstance(project_section["dependencies"], list):
                        raise ValueError(
                            "Invalid [project.dependencies] format. "
                            "Dependencies must be a sequence/array, never a map or dictionary."
                        )
                    if "dependency-groups" in project_section:
                        raise ValueError(
                            "Forbidden 'dependency-groups' key inside [project]. "
                            "Define [dependency-groups] as its own top-level table."
                        )

                dep_groups = parsed_toml.get("dependency-groups")
                if dep_groups is not None:
                    if not isinstance(dep_groups, dict) or any(
                        not isinstance(v, list) or not all(isinstance(item, str) for item in v)
                        for v in dep_groups.values()
                    ):
                        raise ValueError(
                            "Invalid [dependency-groups] format. Each group must be an array of dependency strings."
                        )

            elif ext in (".json",):
                # Check JSON formatting
                json.loads(content)

        except Exception as e:
            validation_error = str(e)

        # If syntax is invalid, reject the write and feed the error back to the Coder agent
        if validation_error:
            return (
                f"❌ PRE-FLIGHT VALIDATION FAILED: Cannot write '{file_path}' due to syntax/parse errors.\n"
                f"Parser Error Details: {validation_error}\n"
                f"Please fix the syntax error in your content and try writing the file again."
            )
        # ----------------------------------------------------

        # If valid, proceed to save the file safely at target_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content, encoding="utf-8")

        if target_path.name == "pyproject.toml":
            workspace_warnings = validate_uv_project(self.workspace_root)
            if workspace_warnings:
                warning_block = "\n".join(f"- {w}" for w in workspace_warnings)
                return (
                    f"Successfully wrote and validated {file_path}\n"
                    f"⚠️ WORKSPACE WARNINGS FOR `uv sync` (resolve before finishing):\n{warning_block}"
                )

        return f"Successfully wrote and validated {file_path}"

    def read_file(self, file_path: str) -> str:
        """Read and return utf-8 text content from a file."""
        if not str(file_path).strip():
            return "Error: File path cannot be empty."

        try:
            target_path = self._resolve_path(file_path)
        except PermissionError as e:
            return f"Error: {e}"

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

        try:
            target_path = self._resolve_path(dir_path)
        except PermissionError as e:
            return f"Error: {e}"
        try:
            target_path.mkdir(parents=True, exist_ok=True)
            return f"Directory created: '{target_path.relative_to(self.workspace_root)}'"
        except Exception as e:
            return f"Failed to create directory '{dir_path}': {str(e)}"

    def list_files(self, relative_dir: str = ".") -> str:
        """List relative paths of all files in directory, ignoring hidden/temp folders."""
        try:
            target_dir = self._resolve_path(relative_dir)
        except PermissionError as e:
            return f"Error: {e}"

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

        try:
            target_path = self._resolve_path(file_path)
        except PermissionError as e:
            return f"Error: {e}"

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
    """Public tool wrapper to delete unnecessary files."""
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