"""Unit tests for CodeEditor pre-flight validation features."""

import pytest
from pathlib import Path
from my_coding_agent.tools.code_editor import CodeEditor


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