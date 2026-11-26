"""Offline mocked smoke and unit tests for CLI entry points and workflow orchestration."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import main as root_main
import my_coding_agent
from my_coding_agent import cli
from my_coding_agent.config import Config
from my_coding_agent.tools.terminal import ExecutionResult


def test_package_entry_point_delegates_to_cli(mocker) -> None:
    """Verify my_coding_agent.main() and main.main() both delegate to cli.run_agent_workflow()."""
    mock_workflow = mocker.patch("my_coding_agent.cli.run_agent_workflow")
    my_coding_agent.main()
    root_main.main()
    assert mock_workflow.call_count == 2


def test_parse_plan_steps_variants() -> None:
    """Verify parse_plan_steps safely handles dicts, lists, fences, invalid JSON, and non-dict items."""
    assert cli.parse_plan_steps([{"id": 1}, "bad_item"]) == [{"id": 1}]
    assert cli.parse_plan_steps({"steps": [{"id": 2}, 42]}) == [{"id": 2}]
    assert cli.parse_plan_steps({"steps": "not-a-list"}) == []

    fenced = '```json\n{"steps": [{"id": 3, "command_to_execute": "uv sync"}]}\n```'
    assert cli.parse_plan_steps(fenced) == [{"id": 3, "command_to_execute": "uv sync"}]
    assert cli.parse_plan_steps("not valid json") == []


def test_run_agent_workflow_happy_path_smoke(tmp_path: Path, mocker) -> None:
    """Dry-run smoke test of run_agent_workflow() with all external LLM/terminal calls mocked."""
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()

    mocker.patch.object(Config, "OBSIDIAN_VAULT_PATH", vault_dir)
    mocker.patch.object(Config, "MAX_RETRIES", 2)
    mocker.patch.object(Config, "validate", return_value=None)
    mocker.patch("my_coding_agent.cli.setup_file_logging")
    mocker.patch("my_coding_agent.cli.Prompt.ask", return_value="Build a sample CLI tool")

    mock_planner = MagicMock()
    mock_planner.suggest_project_structure.return_value = {
        "folder_name": "sample_cli",
        "main_file": "main.py",
    }
    mock_planner.analyze_verification_strategy.return_value = {
        "target_file": "main.py",
        "test_file": "tests/test_main.py",
        "test_command": "pytest tests/test_main.py",
        "mode": "unit_test",
    }
    mock_planner.create_plan.return_value = json.dumps(
        {
            "steps": [
                {"id": 1, "description": "Sync deps", "command_to_execute": "uv sync"},
                {"id": 2, "description": "Duplicate sync", "command_to_execute": "uv sync"},
                {"id": 3, "description": "Run tests", "command_to_execute": "pytest tests/test_main.py"},
            ]
        }
    )
    mocker.patch("my_coding_agent.cli.PlannerAgent", return_value=mock_planner)

    mock_coder = MagicMock()
    mocker.patch("my_coding_agent.cli.CoderAgent", return_value=mock_coder)

    mock_tester = MagicMock()
    mock_tester.generate_test_code.return_value = "def test_ok():\n    assert 1 + 1 == 2\n"
    mocker.patch("my_coding_agent.cli.TesterAgent", return_value=mock_tester)

    mock_terminal = MagicMock()
    mock_terminal.run.side_effect = [
        ExecutionResult(success=True, returncode=0, stdout="Synced", stderr="", command="uv sync"),
        ExecutionResult(success=True, returncode=0, stdout="1 passed", stderr="", command="pytest tests/test_main.py"),
    ]
    mocker.patch("my_coding_agent.cli.TerminalRunner", return_value=mock_terminal)

    cli.run_agent_workflow(base_projects_dir=tmp_path)

    workspace = tmp_path / "sample_cli"
    assert workspace.is_dir()
    assert (workspace / "tests" / "test_main.py").exists()
    mock_planner.suggest_project_structure.assert_called_once()
    mock_planner.analyze_verification_strategy.assert_called_once_with(
        "Build a sample CLI tool", target_file="main.py"
    )
    mock_coder.execute_task.assert_called_once()
    # Setup command deduplicated + main test command = 2 calls
    assert mock_terminal.run.call_count == 2
    assert (vault_dir / "50_Logs" / "AgentLogs").exists()


def test_run_agent_workflow_setup_failure_and_abort_smoke(tmp_path: Path, mocker) -> None:
    """Verify setup failure records execution history, triggers RCA, and aborts cleanly on 'q'."""
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()

    mocker.patch.object(Config, "OBSIDIAN_VAULT_PATH", vault_dir)
    mocker.patch.object(Config, "MAX_RETRIES", 1)
    mocker.patch.object(Config, "validate", return_value=None)
    mocker.patch("my_coding_agent.cli.setup_file_logging")
    mocker.patch("my_coding_agent.cli.Prompt.ask", side_effect=["Build broken project", "q"])

    mock_planner = MagicMock()
    mock_planner.suggest_project_structure.return_value = {
        "folder_name": "broken_proj",
        "main_file": "main.py",
    }
    mock_planner.analyze_verification_strategy.return_value = {
        "target_file": "main.py",
        "test_file": None,
        "test_command": "python main.py",
        "mode": "direct_execution",
    }
    mock_planner.create_plan.return_value = json.dumps(
        {"steps": [{"id": 1, "command_to_execute": "uv sync"}]}
    )
    mocker.patch("my_coding_agent.cli.PlannerAgent", return_value=mock_planner)
    mocker.patch("my_coding_agent.cli.CoderAgent", return_value=MagicMock())

    mock_tester = MagicMock()
    mock_tester.execute_task.return_value = "Root cause: invalid pyproject.toml"
    mocker.patch("my_coding_agent.cli.TesterAgent", return_value=mock_tester)

    mock_terminal = MagicMock()
    mock_terminal.run.return_value = ExecutionResult(
        success=False, returncode=1, stdout="", stderr="TOML parse error", command="uv sync"
    )
    mocker.patch("my_coding_agent.cli.TerminalRunner", return_value=mock_terminal)

    mock_obsidian = MagicMock()
    mock_obsidian.read_rules.return_value = ""
    mocker.patch("my_coding_agent.cli.ObsidianAdapter", return_value=mock_obsidian)

    cli.run_agent_workflow(base_projects_dir=tmp_path)

    mock_tester.execute_task.assert_called_once()
    mock_tester.generate_obsidian_troubleshooting_note.assert_called_once()
    mock_obsidian.append_log.assert_called_once()
    call_kwargs = mock_obsidian.append_log.call_args.kwargs
    assert call_kwargs["status"] == "ABORTED_BY_USER"
    assert len(call_kwargs["execution_history"]) == 1
    assert "TOML parse error" in call_kwargs["execution_history"][0]["output"]


def test_run_agent_workflow_empty_prompt_exits_cleanly(tmp_path: Path, mocker) -> None:
    """Verify empty task prompt exits immediately without running agents."""
    mocker.patch.object(Config, "validate", return_value=None)
    mocker.patch("my_coding_agent.cli.setup_file_logging")
    mocker.patch("my_coding_agent.cli.Prompt.ask", return_value="   ")
    planner_cls = mocker.patch("my_coding_agent.cli.PlannerAgent")

    cli.run_agent_workflow(base_projects_dir=tmp_path)
    planner_cls.assert_not_called()


def test_run_agent_workflow_missing_api_keys_exits(tmp_path: Path, mocker) -> None:
    """Verify missing API credentials abort before prompting for task input."""
    mocker.patch.object(Config, "validate", side_effect=ValueError("No API keys found."))
    mocker.patch("my_coding_agent.cli.setup_file_logging")
    prompt_mock = mocker.patch("my_coding_agent.cli.Prompt.ask")

    with pytest.raises(SystemExit) as exc_info:
        cli.run_agent_workflow(base_projects_dir=tmp_path)

    assert exc_info.value.code == 1
    prompt_mock.assert_not_called()


def test_run_agent_workflow_preflight_blocks_broken_hatchling_before_uv_sync(tmp_path: Path, mocker) -> None:
    """Verify cli pre-flight catches missing README.md & hatchling package before invoking uv sync."""
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()

    mocker.patch.object(Config, "OBSIDIAN_VAULT_PATH", vault_dir)
    mocker.patch.object(Config, "MAX_RETRIES", 1)
    mocker.patch.object(Config, "validate", return_value=None)
    mocker.patch("my_coding_agent.cli.setup_file_logging")
    mocker.patch("my_coding_agent.cli.Prompt.ask", side_effect=["Build slugify module", "q"])

    mock_planner = MagicMock()
    mock_planner.suggest_project_structure.return_value = {
        "folder_name": "slugify_module",
        "main_file": "slugify.py",
    }
    mock_planner.analyze_verification_strategy.return_value = {
        "target_file": "slugify.py",
        "test_file": None,
        "test_command": "pytest",
        "mode": "unit_test",
    }
    mock_planner.create_plan.return_value = json.dumps(
        {"steps": [{"id": 1, "command_to_execute": "uv sync"}]}
    )
    mocker.patch("my_coding_agent.cli.PlannerAgent", return_value=mock_planner)

    def write_broken_workspace(**_kwargs):
        ws = tmp_path / "slugify_module"
        (ws / "pyproject.toml").write_text(
            '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n\n'
            '[project]\nname = "slugify-module"\nversion = "0.1.0"\nreadme = "README.md"\n',
            encoding="utf-8",
        )
        (ws / "slugify.py").write_text("def to_slug(s): return s\n", encoding="utf-8")

    mock_coder = MagicMock()
    mock_coder.execute_task.side_effect = write_broken_workspace
    mocker.patch("my_coding_agent.cli.CoderAgent", return_value=mock_coder)

    mock_tester = MagicMock()
    mock_tester.execute_task.return_value = "Missing README.md and package dir"
    mocker.patch("my_coding_agent.cli.TesterAgent", return_value=mock_tester)

    mock_terminal = MagicMock()
    mocker.patch("my_coding_agent.cli.TerminalRunner", return_value=mock_terminal)

    cli.run_agent_workflow(base_projects_dir=tmp_path)

    # Pre-flight should have blocked `uv sync` and sent both issues straight to RCA
    mock_terminal.run.assert_not_called()
    mock_tester.execute_task.assert_called_once()
    rca_log = mock_tester.execute_task.call_args.kwargs["execution_log"]
    assert "README.md" in rca_log
    assert "slugify_module/" in rca_log
