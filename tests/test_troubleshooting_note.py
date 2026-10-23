from pathlib import Path
from unittest.mock import patch
import pytest
import yaml

from my_coding_agent.agents.tester import TesterAgent


@pytest.fixture
def tester_agent(mocker):
    """Fixture that initializes a real TesterAgent instance with mocked external dependencies."""
    agent = TesterAgent(
        workspace_root=Path("./dummy_workspace"),
        obsidian_vault_path=Path("./dummy_vault")
    )
    
    # Mock the UI logger so it doesn't print to the console during tests
    mocker.patch.object(agent.ui_logger, "info")
    mocker.patch.object(agent.ui_logger, "status")
    mocker.patch.object(agent.ui_logger, "stop_action")
    
    # Mock the obsidian adapter so we can control or track saves
    mocker.patch.object(agent, "obsidian_adapter")
    
    return agent


def test_generate_obsidian_troubleshooting_note_hybrid_structure(tester_agent, mocker):
    """Verify that LLM-extracted metadata is safely parsed and structured into valid YAML frontmatter."""
    
    # Mock the structured Pydantic response that call_llm_structured returns
    from my_coding_agent.agents.tester import TroubleshootingAnalysis
    mock_structured_data = TroubleshootingAnalysis(
        tags=["python", "virtualenv", "uv-sync"],
        related_topics=["Python", "Virtual Environment"],
        root_cause_analysis="The dynamic link failed due to missing environment variables.",
        solution_and_prevention="1. Ensure environment variables are loaded via `.env` file."
    )
    
    mocker.patch.object(tester_agent, "call_llm_structured", return_value=mock_structured_data)

    error_title = "ModuleNotFoundError in pytest execution"
    error_log = "ModuleNotFoundError: No module named 'tester'"
    solution_summary = "Added PYTHONPATH setting to pytest context."

    # Execute the actual instance method directly!
    result_md = tester_agent.generate_obsidian_troubleshooting_note(
        error_title=error_title,
        error_log=error_log,
        solution_summary=solution_summary,
        save_to_vault=True,
    )

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

    # 4. Verify clean separation of markdown body
    assert "## 1. Symptom & Error Log" in result_md
    assert "## 2. Root Cause Analysis" in result_md
    assert "## 3. Solution & Prevention Rules" in result_md

    # 5. Verify vault adapter delegation
    tester_agent.obsidian_adapter.save_troubleshooting_note.assert_called_once_with(
        title=error_title,
        content=result_md,
        relative_folder="30_Resources/Troubleshooting",
    )


def test_generate_obsidian_troubleshooting_note_yaml_validity(tester_agent, tmp_path, mocker):
    """Verify that generated markdown contains strictly valid YAML frontmatter that parses without errors."""
    from my_coding_agent.agents.tester import TroubleshootingAnalysis
    mock_structured_data = TroubleshootingAnalysis(
        tags=["python", "pytest", "yaml-parsing"],
        related_topics=["Python", "PyYAML", "Obsidian Vault"],
        root_cause_analysis="Invalid indentation in YAML block caused Obsidian parsing failure.",
        solution_and_prevention="1. Construct YAML strings using explicit lines."
    )
    
    mocker.patch.object(tester_agent, "call_llm_structured", return_value=mock_structured_data)

    # Use a real file system mock using pytest's tmp_path fixture
    def fake_save(title, content, relative_folder):
        target_dir = tmp_path / relative_folder
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / "2026-09-30_test_issue.md"
        file_path.write_text(content, encoding="utf-8")
        return file_path

    tester_agent.obsidian_adapter.save_troubleshooting_note.side_effect = fake_save

    result_md = tester_agent.generate_obsidian_troubleshooting_note(
        error_title="YAML Frontmatter Parsing Error",
        error_log="ScannerError: mapping values are not allowed here",
        solution_summary="Fixed multiline string formatting in Python builder.",
        save_to_vault=True,
    )

    # 1. Extract YAML block between '---' markers
    assert result_md.startswith("---\n")
    parts = result_md.split("---\n", 2)
    yaml_text = parts[1]

    # 2. Parse extracted YAML text with PyYAML
    parsed_yaml = yaml.safe_load(yaml_text)

    # 3. Assert structure of parsed YAML content
    assert isinstance(parsed_yaml, dict)
    assert parsed_yaml["type"] == "troubleshooting"
    assert "python" in parsed_yaml["tags"]
    assert "[[PyYAML]]" in parsed_yaml["related_topics"]

    # 4. Verify actual file written to disk
    saved_file = tmp_path / "30_Resources/Troubleshooting/2026-09-30_test_issue.md"
    assert saved_file.exists()
    assert saved_file.read_text(encoding="utf-8") == result_md