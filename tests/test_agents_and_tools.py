"""Offline unit tests for Config, Agents (Base, Planner, Coder, Tester), and Tools."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from my_coding_agent.agents.base import BaseAgent, extract_response_content
from my_coding_agent.agents.coder import CoderAgent
from my_coding_agent.agents.planner import PlannerAgent
from my_coding_agent.agents.tester import TesterAgent
from my_coding_agent.config import Config
from my_coding_agent.tools.code_editor import CodeEditor
from my_coding_agent.tools.obsidian import ObsidianAdapter
from my_coding_agent.tools.terminal import TerminalRunner


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------


def test_config_normalize_and_fallbacks(monkeypatch) -> None:
    """Verify alias normalization and environment override handling in Config."""
    assert Config.normalize_model_name("gpt-4o-mini") == "openrouter/openai/gpt-4o-mini"
    assert Config.normalize_model_name("deepseek-chat") == "openrouter/deepseek/deepseek-chat"
    assert Config.normalize_model_name("  ") == ""

    monkeypatch.setenv("MODEL_FALLBACK_LIST", "gpt-4o, groq/llama-3.3-70b-versatile")
    fallbacks = Config.get_model_fallbacks()
    assert fallbacks == ["openrouter/openai/gpt-4o", "groq/llama-3.3-70b-versatile"]


def test_config_validate_requires_key(mocker) -> None:
    """Verify Config.validate raises ValueError when all API keys are empty."""
    mocker.patch.object(Config, "OPENROUTER_API_KEY", "")
    mocker.patch.object(Config, "GROQ_API_KEY", "")
    mocker.patch.object(Config, "GEMINI_API_KEY", "")
    mocker.patch.object(Config, "OPENAI_API_KEY", "")

    with pytest.raises(ValueError, match="No API keys found"):
        Config.validate()


# ---------------------------------------------------------------------------
# BaseAgent Tests
# ---------------------------------------------------------------------------


class SampleSchema(BaseModel):
    answer: str


def test_extract_response_content_handles_none_tool_call_message() -> None:
    """Verify extract_response_content returns '' when message.content is None."""
    msg = SimpleNamespace(content=None, tool_calls=[{"id": "1"}])
    resp = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
    assert extract_response_content(resp) == ""
    assert extract_response_content(msg) == ""


def test_base_agent_fallback_and_structured_output(mocker) -> None:
    """Verify BaseAgent falls back across models and parses fenced JSON into Pydantic."""
    mocker.patch("time.sleep")
    mocker.patch.object(Config, "GEMINI_API_KEY", "")
    agent = BaseAgent(
        name="Tester",
        system_prompt="System",
        model_fallback_list=["openai/gpt-4o-mini", "groq/llama-3.3-70b-versatile"],
    )

    fenced_resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='```json\n{"answer": "ok"}\n```'))]
    )
    completion_mock = mocker.patch(
        "litellm.completion",
        side_effect=[RuntimeError("rate limit"), fenced_resp],
    )

    parsed = agent.call_llm_structured(
        messages=[{"role": "user", "content": "hi"}],
        response_format=SampleSchema,
    )
    assert parsed.answer == "ok"
    assert completion_mock.call_count == 2


def test_base_agent_gemini_native_rejects_openai_tools_and_falls_back(mocker) -> None:
    """Verify Gemini native path refuses OpenAI tool dicts so call_llm falls back cleanly."""
    mocker.patch("time.sleep")
    mocker.patch.object(Config, "GEMINI_API_KEY", "fake-key")
    mocker.patch("google.genai.Client", return_value=MagicMock())

    agent = BaseAgent(
        name="CustomAgent",
        system_prompt="System",
        model_fallback_list=["gemini-2.5-flash", "groq/llama-3.3-70b-versatile"],
    )
    ok_resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="fallback ok", tool_calls=None))]
    )
    mocker.patch("litellm.completion", return_value=ok_resp)

    res = agent.call_llm(
        messages=[{"role": "user", "content": "write file"}],
        tools=[{"type": "function", "function": {"name": "write_file"}}],
    )
    assert extract_response_content(res) == "fallback ok"


# ---------------------------------------------------------------------------
# PlannerAgent, CoderAgent, TesterAgent Tests
# ---------------------------------------------------------------------------


def test_planner_skips_duplicate_structure_call_and_normalizes_list_plan(tmp_path: Path, mocker) -> None:
    """Verify PlannerAgent reuses provided target_file and wraps list JSON plans into steps."""
    planner = PlannerAgent(workspace_root=tmp_path, obsidian_vault_path=tmp_path)
    suggest_spy = mocker.spy(planner, "suggest_project_structure")

    strategy_resp = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"target_file": "app.py", "test_file": "tests/test_app.py", "test_command": "pytest", "mode": "unit_test"}'
                )
            )
        ]
    )
    list_plan_resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='[{"id": 1, "description": "step"}]'))]
    )
    mocker.patch.object(planner, "call_llm", side_effect=[strategy_resp, list_plan_resp])

    strategy = planner.analyze_verification_strategy("Build app", target_file="app.py")
    assert strategy["target_file"] == "app.py"
    suggest_spy.assert_not_called()

    plan_json = planner.create_plan("Build app", target_file="app.py", auto_fetch_vault_rules=False)
    assert json.loads(plan_json) == {"steps": [{"id": 1, "description": "step"}]}


def test_coder_extract_pure_code_preserves_markdown_with_inner_fences(tmp_path: Path) -> None:
    """Verify CoderAgent does not truncate README.md files containing inner code blocks."""
    coder = CoderAgent(workspace_root=tmp_path, obsidian_vault_path=tmp_path)

    md_content = "# Title\n\n```bash\nuv run pytest\n```\n\nDone."
    tool_call = SimpleNamespace(
        function=SimpleNamespace(
            name="write_file",
            arguments=json.dumps({"file_path": "README.md", "content": md_content}),
        )
    )
    result = coder._execute_tool_call(tool_call)
    assert "Successfully wrote" in result
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == md_content

    py_fenced = "```python\nx = 1\n```"
    assert coder.extract_pure_code(py_fenced) == "x = 1"


def test_tester_execute_task_returns_string_when_iterations_exhausted(tmp_path: Path, mocker) -> None:
    """Verify TesterAgent.execute_task returns extracted text even at max_tool_iterations."""
    tester = TesterAgent(workspace_root=tmp_path, obsidian_vault_path=tmp_path)
    tool_call = SimpleNamespace(
        id="tc_1",
        function=SimpleNamespace(name="search_files", arguments='{"keyword": ".py"}'),
    )
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Final RCA text", tool_calls=[tool_call]))]
    )
    mocker.patch.object(tester, "call_llm", return_value=resp)

    out = tester.execute_task("Diagnose failure", max_tool_iterations=1, auto_fetch_vault_rules=False)
    assert out == "Final RCA text"


# ---------------------------------------------------------------------------
# Tools Tests (CodeEditor, TerminalRunner, ObsidianAdapter)
# ---------------------------------------------------------------------------


def test_code_editor_security_and_crud(tmp_path: Path) -> None:
    """Verify CodeEditor blocks path traversal and empty paths while handling CRUD cleanly."""
    editor = CodeEditor(workspace_root=str(tmp_path))

    assert "Access denied" in editor.write_file("../escape.py", "x = 1")
    assert not (tmp_path.parent / "escape.py").exists()
    assert not (tmp_path / "escape.py").exists()

    assert "cannot be empty" in editor.write_file("  ", "x = 1")
    assert "Successfully wrote" in editor.write_file("pkg/mod.py", "VAL = 42\n")
    assert "VAL = 42" in editor.read_file("pkg/mod.py")
    assert "pkg/mod.py" in editor.search_files("mod")
    assert "Successfully deleted" in editor.delete_file("pkg/mod.py")
    assert "does not exist" in editor.read_file("pkg/mod.py")


def test_terminal_runner_env_isolation_empty_and_timeout(mocker) -> None:
    """Verify TerminalRunner isolates parent venv, rejects empty commands, and decodes timeout bytes."""
    runner = TerminalRunner(default_timeout=5)
    mocker.patch.dict(
        "os.environ",
        {"VIRTUAL_ENV": "/tmp/fake_venv", "PYTHONHOME": "/tmp/pyhome", "PATH": "/tmp/fake_venv/bin:/usr/bin"},
        clear=False,
    )
    isolated = runner._get_isolated_env()
    assert "VIRTUAL_ENV" not in isolated
    assert "PYTHONHOME" not in isolated
    assert "/tmp/fake_venv/bin" not in isolated["PATH"]

    empty_res = runner.run("   ")
    assert empty_res.success is False
    assert "cannot be empty" in empty_res.stderr

    mocker.patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="sleep 10", timeout=1, output=b"partial output"),
    )
    timeout_res = runner.run("sleep 10", timeout=1)
    assert timeout_res.success is False
    assert timeout_res.stdout == "partial output"
    assert "timed out" in timeout_res.stderr


def test_obsidian_adapter_rules_search_and_logs(tmp_path: Path) -> None:
    """Verify ObsidianAdapter read_rules safety, frontmatter search scoring, and log writing."""
    missing_vault = tmp_path / "does_not_exist"
    assert ObsidianAdapter(str(missing_vault)).read_rules("build api") == "No specific rules found."
    assert not missing_vault.exists()

    vault = tmp_path / "vault"
    rules_dir = vault / "30_Resources" / "AgentRules"
    rules_dir.mkdir(parents=True)
    (rules_dir / "general_rules.md").write_text("Always use type hints.", encoding="utf-8")
    (rules_dir / "http_requests.md").write_text("Use timeouts on HTTP calls.", encoding="utf-8")

    # Body-only note with horizontal rules should NOT get frontmatter score bonus
    notes_dir = vault / "30_Resources" / "Notes"
    notes_dir.mkdir(parents=True)
    (notes_dir / "plain_note.md").write_text(
        "# Plain Note\n---\nfastapi guide\n---\nfastapi details",
        encoding="utf-8",
    )
    archive_dir = vault / "40_Archives"
    archive_dir.mkdir(parents=True)
    (archive_dir / "old_fastapi.md").write_text("fastapi archived", encoding="utf-8")

    adapter = ObsidianAdapter(str(vault))
    rules_text = adapter.read_rules("Build a REST API endpoint")
    assert "Always use type hints." in rules_text
    assert "Use timeouts on HTTP calls." in rules_text

    content, sources = adapter.search_with_sources(["fastapi"])
    assert len(sources) == 1
    assert "plain_note.md (Score: 2)" in sources[0]
    assert "old_fastapi.md" not in content

    log_path = adapter.append_log(
        task_description="Test task",
        status="SUCCESS",
        execution_history=[{"attempt": 1, "success": True, "output": "Passed"}],
    )
    assert log_path.exists()
    assert "SUCCESS" in log_path.read_text(encoding="utf-8")


def test_obsidian_search_preserves_code_blocks_around_matches(tmp_path: Path) -> None:
    """Verify ObsidianAdapter includes contextual code blocks near matches in long guides."""
    vault = tmp_path / "vault"
    guide_dir = vault / "30_Resources" / "ProjectSchemas"
    guide_dir.mkdir(parents=True)

    filler = "\n".join(f"Line {i}: general documentation text." for i in range(65))
    guide_text = (
        "## pyproject.toml Template\n"
        "```toml\n"
        "[build-system]\n"
        'requires = ["hatchling"]\n'
        'build-backend = "hatchling.build"\n'
        "```\n\n"
        + filler
    )
    (guide_dir / "Guide.md").write_text(guide_text, encoding="utf-8")

    adapter = ObsidianAdapter(str(vault))
    content, _ = adapter.search_with_sources(["pyproject.toml"])
    assert 'build-backend = "hatchling.build"' in content


def test_ui_render_failure_panel_preserves_traceback_tail(capsys) -> None:
    """Verify ui.render_failure_panel keeps the tail of long tracebacks where root errors live."""
    from my_coding_agent.utils import ui

    long_traceback = (
        "Building slugify-module\n"
        + ("  File 'hatchling/build.py', line 83\n" * 40)
        + "OSError: Readme file does not exist: README.md"
    )
    ui.render_failure_panel(long_traceback, 1)
    captured = capsys.readouterr().out
    assert "OSError: Readme file does not exist: README.md" in captured
