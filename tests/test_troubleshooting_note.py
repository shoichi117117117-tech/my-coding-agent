from unittest.mock import MagicMock, patch
import pytest
import yaml


@pytest.fixture
def mock_agent():
    """Fixture to create a mock instance of the troubleshooting worker agent."""
    agent = MagicMock()

    # Mock logger to prevent UI side effects during tests
    agent.ui_logger = MagicMock()
    agent.obsidian_adapter = MagicMock()

    return agent


def test_generate_obsidian_troubleshooting_note_hybrid_structure(mock_agent):
    """Verify that LLM-extracted metadata is safely parsed and structured into valid YAML frontmatter."""
    # Mock LLM response containing dynamic tags, wiki links, and raw markdown body
    mock_llm_response = (
        "TAGS: python, virtualenv, uv-sync\n"
        "RELATED_TOPICS: [[Python]], [[Virtual Environment]]\n\n"
        "## 2. Root Cause Analysis\n"
        "The dynamic link failed due to missing environment variables.\n\n"
        "## 3. Solution & Prevention Rules\n"
        "1. Ensure environment variables are loaded via `.env` file."
    )

    # Attach mock LLM call implementation
    mock_agent.call_llm.return_value = mock_llm_response

    # Bind the target method to the mock agent
    from agents.tester import generate_obsidian_troubleshooting_note

    mock_agent.generate_obsidian_troubleshooting_note = (
        generate_obsidian_troubleshooting_note.__get__(
            mock_agent, type(mock_agent)
        )
    )

    # Execute target function
    error_title = "ModuleNotFoundError in pytest execution"
    error_log = "ModuleNotFoundError: No module named 'tester'"
    solution_summary = "Added PYTHONPATH setting to pytest context."

    result_md = mock_agent.generate_obsidian_troubleshooting_note(
        error_title=error_title,
        error_log=error_log,
        solution_summary=solution_summary,
        save_to_vault=True,
    )

    # ------------------------------------------------------------------
    # Assertions: Verify Hybrid Architecture Outputs
    # ------------------------------------------------------------------

    # 1. Verify strict YAML Frontmatter boundaries
    assert result_md.startswith("---\n")
    assert "\n---\n\n" in result_md

    # 2. Verify dynamic tag parsing and sanitization
    assert "  - troubleshooting" in result_md
    assert "  - error-solution" in result_md
    assert "  - python" in result_md
    assert "  - virtualenv" in result_md
    assert "  - uv-sync" in result_md

    # 3. Verify Obsidian WikiLink topic parsing
    assert '  - "[[Python]]"' in result_md
    assert '  - "[[Virtual Environment]]"' in result_md

    # 4. Verify clean separation of markdown body (meta lines removed)
    assert "TAGS:" not in result_md
    assert "RELATED_TOPICS:" not in result_md
    assert "## 1. Symptom & Error Log" in result_md
    assert "## 2. Root Cause Analysis" in result_md
    assert "## 3. Solution & Prevention Rules" in result_md

    # 5. Verify vault adapter delegation
    mock_agent.obsidian_adapter.save_troubleshooting_note.assert_called_once_with(
        title=error_title,
        content=result_md,
        relative_folder="30_Resources/Troubleshooting",
    )


def test_generate_obsidian_troubleshooting_note_markdown_fence_stripping(
    mock_agent,
):
    """Verify that redundant markdown code fences surrounding LLM output are cleanly stripped."""
    # Mock LLM response wrapped accidentally in markdown triple backticks
    mock_llm_response = (
        "```markdown\n"
        "TAGS: fastapi, async\n"
        "RELATED_TOPICS: [[FastAPI]]\n\n"
        "## 2. Root Cause Analysis\n"
        "Unhandled exception in async loop.\n"
        "```"
    )

    mock_agent.call_llm.return_value = mock_llm_response

    from agents.tester import generate_obsidian_troubleshooting_note

    mock_agent.generate_obsidian_troubleshooting_note = (
        generate_obsidian_troubleshooting_note.__get__(
            mock_agent, type(mock_agent)
        )
    )

    result_md = mock_agent.generate_obsidian_troubleshooting_note(
        error_title="Async Timeout",
        error_log="TimeoutError",
        solution_summary="Increased timeout threshold.",
        save_to_vault=False,
    )

    # Ensure outer ```markdown wrappers were completely stripped
    assert not result_md.startswith("```markdown")
    assert result_md.startswith("---\n")
    assert "  - fastapi" in result_md
    assert '  - "[[FastAPI]]"' in result_md

def test_generate_obsidian_troubleshooting_note_yaml_validity(mock_agent, tmp_path):
    """Verify that generated markdown contains strictly valid YAML frontmatter that parses without errors."""
    mock_llm_response = (
        "TAGS: python, pytest, yaml-parsing\n"
        "RELATED_TOPICS: [[Python]], [[PyYAML]], [[Obsidian Vault]]\n\n"
        "## 2. Root Cause Analysis\n"
        "Invalid indentation in YAML block caused Obsidian parsing failure.\n\n"
        "## 3. Solution & Prevention Rules\n"
        "1. Construct YAML strings using explicit lines without textwrap.dedent side-effects."
    )

    mock_agent.call_llm.return_value = mock_llm_response

    # Use a real file system write mock using pytest's tmp_path fixture
    def fake_save(title, content, relative_folder):
        target_dir = tmp_path / relative_folder
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / "2026-09-30_test_issue.md"
        file_path.write_text(content, encoding="utf-8")
        return file_path

    mock_agent.obsidian_adapter.save_troubleshooting_note.side_effect = fake_save

    from agents.tester import generate_obsidian_troubleshooting_note

    mock_agent.generate_obsidian_troubleshooting_note = (
        generate_obsidian_troubleshooting_note.__get__(
            mock_agent, type(mock_agent)
        )
    )

    # Execute note generation
    result_md = mock_agent.generate_obsidian_troubleshooting_note(
        error_title="YAML Frontmatter Parsing Error",
        error_log="ScannerError: mapping values are not allowed here",
        solution_summary="Fixed multiline string formatting in Python builder.",
        save_to_vault=True,
    )

    # 1. Extract YAML block between '---' markers
    assert result_md.startswith("---\n")
    parts = result_md.split("---\n", 2)
    assert len(parts) >= 3, "Markdown must contain opening and closing '---' frontmatter dividers."

    yaml_text = parts[1]

    # 2. Parse extracted YAML text with PyYAML
    parsed_yaml = yaml.safe_load(yaml_text)

    # 3. Assert structure and exact types of parsed YAML content
    assert isinstance(parsed_yaml, dict), "Frontmatter must parse into a valid dictionary."
    assert parsed_yaml["type"] == "troubleshooting"
    assert "date" in parsed_yaml

    # Verify tags parsed as list
    assert isinstance(parsed_yaml["tags"], list)
    assert "troubleshooting" in parsed_yaml["tags"]
    assert "python" in parsed_yaml["tags"]
    assert "pytest" in parsed_yaml["tags"]

    # Verify WikiLinks parsed cleanly as list strings
    assert isinstance(parsed_yaml["related_topics"], list)
    assert "[[Python]]" in parsed_yaml["related_topics"]
    assert "[[PyYAML]]" in parsed_yaml["related_topics"]

    # 4. Verify actual file written to disk contains identical content
    saved_file = tmp_path / "30_Resources/Troubleshooting/2026-09-30_test_issue.md"
    assert saved_file.exists()
    assert saved_file.read_text(encoding="utf-8") == result_md