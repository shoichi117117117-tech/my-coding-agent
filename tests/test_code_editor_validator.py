"""Unit tests for CodeEditor pre-flight validation features."""

import pytest
from pathlib import Path
from my_coding_agent.tools.code_editor import CodeEditor, validate_uv_project


@pytest.fixture
def editor(tmp_path: Path) -> CodeEditor:
    """Initialize CodeEditor pointing strictly to pytest's temporary test directory."""
    return CodeEditor(workspace_root=tmp_path)


def test_write_valid_python(editor: CodeEditor) -> None:
    """Verify valid Python files pass pre-flight validation and save inside workspace."""
    code_content = "def add(a, b):\n    return a + b\n"
    result = editor.write_file("valid_script.py", code_content)
    
    assert "Successfully wrote" in result
    assert (editor.workspace_root / "valid_script.py").read_text(encoding="utf-8") == code_content


def test_write_invalid_python(editor: CodeEditor) -> None:
    """Verify invalid Python syntax triggers pre-flight validation failure and blocks writing."""
    bad_code = "def add(a, b)\n    return a + b\n"  # Missing colon
    result = editor.write_file("bad_script.py", bad_code)
    
    assert "PRE-FLIGHT VALIDATION FAILED" in result
    assert not (editor.workspace_root / "bad_script.py").exists()


def test_write_valid_toml(editor: CodeEditor) -> None:
    """Verify valid pyproject.toml PEP 621 format passes validation."""
    valid_toml = (
        '[project]\n'
        'name = "test-proj"\n'
        'version = "0.1.0"\n'
        'dependencies = [\n'
        '    "pytest>=8.0.0",\n'
        ']\n'
    )
    result = editor.write_file("pyproject.toml", valid_toml)
    
    assert "Successfully wrote" in result
    assert (editor.workspace_root / "pyproject.toml").read_text(encoding="utf-8") == valid_toml


def test_write_invalid_toml_syntax(editor: CodeEditor) -> None:
    """Verify malformed TOML syntax triggers pre-flight failure."""
    bad_toml = (
        '[project\n'  # Unclosed bracket
        'name = "test-proj"\n'
    )
    result = editor.write_file("pyproject.toml", bad_toml)
    
    assert "PRE-FLIGHT VALIDATION FAILED" in result
    assert not (editor.workspace_root / "pyproject.toml").exists()


def test_write_invalid_toml_dependency_map(editor: CodeEditor) -> None:
    """Verify the specific [project.dependencies] map error is caught pre-flight."""
    forbidden_toml = (
        '[project]\n'
        'name = "test-proj"\n\n'
        '[project.dependencies]\n'  # Invalid type: map instead of sequence
        'pytest = "8.0.0"\n'
    )
    result = editor.write_file("pyproject.toml", forbidden_toml)
    
    assert "PRE-FLIGHT VALIDATION FAILED" in result
    assert not (editor.workspace_root / "pyproject.toml").exists()


def test_write_valid_json(editor: CodeEditor) -> None:
    """Verify valid JSON configuration files pass validation."""
    valid_json = '{\n  "name": "test",\n  "version": 1\n}\n'
    result = editor.write_file("config.json", valid_json)
    
    assert "Successfully wrote" in result
    assert (editor.workspace_root / "config.json").exists()


def test_write_invalid_json(editor: CodeEditor) -> None:
    """Verify malformed JSON formatting triggers pre-flight error."""
    bad_json = '{\n  "name": "test",\n  "version": 1,\n}\n'
    result = editor.write_file("config.json", bad_json)
    
    assert "PRE-FLIGHT VALIDATION FAILED" in result
    assert not (editor.workspace_root / "config.json").exists()


def test_write_pyproject_rejects_nested_and_invalid_dependency_groups(editor: CodeEditor) -> None:
    """Verify nested or malformed dependency-groups in pyproject.toml are rejected pre-flight."""
    nested_toml = (
        '[project]\n'
        'name = "demo"\n'
        'version = "0.1.0"\n'
        'dependency-groups = { dev = ["pytest>=8.0.0"] }\n'
    )
    res_nested = editor.write_file("pyproject.toml", nested_toml)
    assert "PRE-FLIGHT VALIDATION FAILED" in res_nested
    assert not (editor.workspace_root / "pyproject.toml").exists()

    bad_groups_toml = (
        '[project]\n'
        'name = "demo"\n'
        'version = "0.1.0"\n\n'
        '[dependency-groups]\n'
        'dev = "pytest>=8.0.0"\n'
    )
    res_bad = editor.write_file("pyproject.toml", bad_groups_toml)
    assert "PRE-FLIGHT VALIDATION FAILED" in res_bad
    assert not (editor.workspace_root / "pyproject.toml").exists()


def test_write_pyproject_warns_missing_readme_and_hatchling_package(editor: CodeEditor) -> None:
    """Verify writing a hatchling pyproject.toml without README.md or package dir emits actionable warnings."""
    hatch_toml = (
        '[build-system]\n'
        'requires = ["hatchling"]\n'
        'build-backend = "hatchling.build"\n\n'
        '[project]\n'
        'name = "slugify-module"\n'
        'version = "0.1.0"\n'
        'readme = "README.md"\n\n'
        '[tool.hatch.build.targets.wheel]\n'
        'packages = ["slugify_module"]\n'
    )
    result = editor.write_file("pyproject.toml", hatch_toml)
    assert "Successfully wrote and validated pyproject.toml" in result
    assert "WORKSPACE WARNINGS FOR `uv sync`" in result
    assert "README.md" in result
    assert "slugify_module" in result


def test_validate_uv_project_slugify_patterns(tmp_path: Path) -> None:
    """Verify validate_uv_project catches both hatchling traps and accepts valid layouts."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[build-system]\n'
        'requires = ["hatchling"]\n'
        'build-backend = "hatchling.build"\n\n'
        '[project]\n'
        'name = "slugify-module"\n'
        'version = "0.1.0"\n'
        'readme = "README.md"\n',
        encoding="utf-8",
    )
    issues = validate_uv_project(tmp_path)
    assert any("Missing readme file 'README.md'" in i for i in issues)
    assert any("slugify_module/" in i for i in issues)

    # Resolving both README.md and package dir clears all issues
    (tmp_path / "README.md").write_text("# slugify-module\n", encoding="utf-8")
    (tmp_path / "slugify_module").mkdir()
    assert validate_uv_project(tmp_path) == []

    # Flat script layout with [tool.uv] package = false is also valid without package dir
    pyproject.write_text(
        '[project]\n'
        'name = "slugify-module"\n'
        'version = "0.1.0"\n\n'
        '[tool.uv]\n'
        'package = false\n',
        encoding="utf-8",
    )
    assert validate_uv_project(tmp_path) == []