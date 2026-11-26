"""CLI entry point and orchestration loop for the autonomous multi-agent development engine."""

import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

from rich.prompt import Prompt

from my_coding_agent.agents.base import extract_response_content
from my_coding_agent.agents.coder import CoderAgent
from my_coding_agent.agents.planner import PlannerAgent
from my_coding_agent.agents.tester import TesterAgent
from my_coding_agent.config import Config
from my_coding_agent.tools.code_editor import CodeEditor, validate_uv_project
from my_coding_agent.tools.obsidian import ObsidianAdapter
from my_coding_agent.tools.terminal import TerminalRunner
from my_coding_agent.utils import ui
from my_coding_agent.utils.logger_config import setup_file_logging

# Global workspace path configuration with fallback default
BASE_PROJECTS_DIR = Path(
    getattr(Config, "BASE_PROJECTS_DIR", Path.home() / "projects_with_agent")
)

# Unified agent loggers
sys_logger = ui.AgentLogger("System")
planner_logger = ui.AgentLogger("Planner")
coder_logger = ui.AgentLogger("Coder")
tester_logger = ui.AgentLogger("Tester")


def parse_plan_steps(plan_output: Any) -> list[dict[str, Any]]:
    """Safely parse structured strategy plan steps from JSON string, dict, or list."""
    raw_steps: Any = []

    if isinstance(plan_output, list):
        raw_steps = plan_output
    elif isinstance(plan_output, dict):
        raw_steps = plan_output.get("steps", [])
    elif isinstance(plan_output, str):
        cleaned = plan_output.strip()
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                raw_steps = parsed.get("steps", [])
            elif isinstance(parsed, list):
                raw_steps = parsed
        except json.JSONDecodeError:
            return []

    if not isinstance(raw_steps, list):
        return []
    return [step for step in raw_steps if isinstance(step, dict)]


def handle_max_retries_exhausted(
    retry_count: int,
    execution_history: list[dict[str, Any]],
    task_instruction: str,
    obsidian: ObsidianAdapter,
) -> tuple[bool, str, int]:
    """Handle human intervention when max retries are exhausted.

    Returns:
        (should_break, updated_formatted_instruction, updated_retry_count)
    """
    if retry_count < Config.MAX_RETRIES:
        return False, task_instruction, retry_count

    ui.render_human_intervention_panel()
    user_feedback = Prompt.ask(
        "[bold yellow]Decision[/bold yellow]", default="r"
    ).strip()

    if user_feedback.lower() == "q":
        sys_logger.warning("Workflow execution aborted by user.")
        obsidian.append_log(
            task_description=task_instruction,
            status="ABORTED_BY_USER",
            execution_history=execution_history,
        )
        return True, task_instruction, retry_count

    updated_instruction = task_instruction
    if user_feedback and user_feedback.lower() != "r":
        updated_instruction += f"\n\n[Human Operator Guidance]: {user_feedback}"

    return False, updated_instruction, 0


def _run_rca_and_archive(
    tester: TesterAgent,
    task_description: str,
    target_file: str,
    error_log: str,
    error_title: str,
) -> str:
    """Run Tester RCA diagnosis, render panel, and archive a troubleshooting note."""
    tester_response = tester.execute_task(
        task_description=task_description,
        target_file=target_file,
        execution_log=error_log,
        auto_fetch_vault_rules=False,
    )
    tester_review = extract_response_content(tester_response)
    ui.render_rca_report_panel(tester_review)

    tester.generate_obsidian_troubleshooting_note(
        error_title=error_title,
        error_log=error_log,
        solution_summary=tester_review,
        save_to_vault=True,
        relative_folder="30_Resources/Troubleshooting",
    )
    return tester_review


def run_agent_workflow(base_projects_dir: Optional[Path] = None) -> None:
    """Execute autonomous coding workflow with closed-loop multi-agent orchestration."""
    setup_file_logging()
    projects_root = Path(base_projects_dir) if base_projects_dir is not None else BASE_PROJECTS_DIR

    ui.render_banner(projects_root, Config.OBSIDIAN_VAULT_PATH, Config.MAX_RETRIES)

    # Validate credentials and initialize adapters safely before prompting
    try:
        Config.validate()
        obsidian = ObsidianAdapter(str(Config.OBSIDIAN_VAULT_PATH))
        terminal = TerminalRunner(default_timeout=30)
    except Exception as e:
        sys_logger.error(f"Failed to initialize system configuration or adapters: {e}")
        sys.exit(1)

    ui.console.print("\n[bold]💡 What would you like to build or refine?[/bold]")
    task_instruction = Prompt.ask("[bold yellow]Task Prompt[/bold yellow]").strip()

    if not task_instruction:
        sys_logger.warning("Task instruction cannot be empty. Terminating workflow.")
        return

    # ---------------------------------------------------------
    # Phase 1: Workspace Isolation & Setup
    # ---------------------------------------------------------
    sys_logger.info("Identifying workspace and verification strategy...")

    planner = PlannerAgent(
        workspace_root=str(projects_root),
        obsidian_vault_path=str(Config.OBSIDIAN_VAULT_PATH),
    )

    project_structure = planner.suggest_project_structure(task_instruction)
    project_folder_name = project_structure.get("folder_name", "generated_project")
    suggested_main_file = project_structure.get("main_file", "main.py")

    project_workspace = projects_root / project_folder_name
    project_workspace.mkdir(parents=True, exist_ok=True)
    workspace_editor = CodeEditor(workspace_root=str(project_workspace))

    formatted_instruction = (
        f"{task_instruction}\n\n"
        "IMPORTANT FOR EXECUTION:\n"
        "1. Do NOT use interactive prompts (e.g., `input()`).\n"
        "2. Ensure valid project manifest files and build/dependency specifications are created if packages are required.\n"
        "3. Provide non-interactive automated verification strategies."
    )

    # ---------------------------------------------------------
    # Phase 2: Dynamic Verification Strategy Inference
    # ---------------------------------------------------------
    obsidian_rules = obsidian.read_rules(task_instruction=task_instruction[:100])
    strategy = planner.analyze_verification_strategy(
        task_instruction, target_file=suggested_main_file
    )

    target_file = str(strategy.get("target_file") or suggested_main_file or "main.py")
    test_file = strategy.get("test_file")
    test_command = strategy.get("test_command")
    execution_mode = str(strategy.get("mode") or "direct_execution")

    # ---------------------------------------------------------
    # Initialize Core Execution Agents
    # ---------------------------------------------------------
    coder = CoderAgent(
        workspace_root=str(project_workspace),
        obsidian_vault_path=str(Config.OBSIDIAN_VAULT_PATH),
    )
    tester = TesterAgent(
        workspace_root=str(project_workspace),
        obsidian_vault_path=str(Config.OBSIDIAN_VAULT_PATH),
    )

    execution_history: list[dict[str, Any]] = []
    retry_count = 0
    current_error: str | None = None
    tester_review: str | None = None

    # ---------------------------------------------------------
    # Phase 3-5: Closed-Loop ReAct / Self-Correction Cycle
    # ---------------------------------------------------------
    while True:
        attempt_num = retry_count + 1

        try:
            # Step A: Planner Generates/Updates Structured Plan
            planner_error_context = None
            if current_error:
                planner_error_context = (
                    f"【Raw Execution Log】\n{current_error}\n\n"
                    f"【Tester Agent RCA Review & Guidance】\n{tester_review if tester_review else 'N/A'}"
                )

            raw_plan_json = planner.create_plan(
                task_instruction=task_instruction,
                target_file=target_file,
                domain_rules=obsidian_rules,
                error_context=planner_error_context,
                attempt=attempt_num,
                auto_fetch_vault_rules=True,
            )
            plan_steps = parse_plan_steps(raw_plan_json)

            # Step B: Tester Generates Test Suite (If applicable)
            if execution_mode == "unit_test" and test_file:
                existing_code = ""
                source_path = project_workspace / target_file
                if source_path.exists():
                    existing_code = source_path.read_text(encoding="utf-8")

                test_code = tester.generate_test_code(
                    target_file=target_file,
                    code_content=existing_code,
                    test_requirements=f"Plan:\n{raw_plan_json}\n\nTask:\n{formatted_instruction}",
                    auto_fetch_vault_rules=False,
                )

                write_res = workspace_editor.write_file(test_file, test_code)
                if "PRE-FLIGHT VALIDATION FAILED" in write_res:
                    tester_logger.warning(f"Generated test suite failed syntax validation: {write_res}")
                else:
                    tester_logger.stop_action(f"Test suite stored at: {test_file}")

            # Step C: Coder Implements/Refactors Codebase
            coder_prompt = (
                "Implement or update the codebase based on the execution plan below.\n"
                "If specific configuration manifests (e.g., pyproject.toml, package.json) are required by the plan, "
                "use `search_obsidian_vault` tool to verify correct schemas if needed.\n\n"
                f"Execution Plan:\n{raw_plan_json}\n\n"
                f"User Instruction:\n{formatted_instruction}"
            )
            coder.execute_task(
                task_description=coder_prompt,
                rules=obsidian_rules,
                error_context=planner_error_context,
                auto_fetch_vault_rules=False,
                iteration_current=attempt_num,
                iteration_total=Config.MAX_RETRIES,
            )

            # Step D: Sequential Plan Execution & Setup Commands (with deduplication)
            plan_execution_failed = False
            executed_commands: set[str] = set()
            for step in plan_steps:
                cmd_to_run = step.get("command_to_execute")
                if cmd_to_run and cmd_to_run not in executed_commands:
                    if not test_command or cmd_to_run.strip() != test_command.strip():
                        executed_commands.add(cmd_to_run)

                        # Pre-flight `uv` / `hatchling` workspace consistency gate before running `uv sync`
                        if re.match(r"^\s*uv\s+sync\b", cmd_to_run):
                            uv_issues = validate_uv_project(project_workspace)
                            if uv_issues:
                                current_error = (
                                    "Pre-flight pyproject.toml / uv sync check failed:\n"
                                    + "\n".join(f"- {issue}" for issue in uv_issues)
                                )
                                coder_logger.error(f"Pre-flight check failed before `{cmd_to_run}`")
                                plan_execution_failed = True
                                execution_history.append(
                                    {
                                        "attempt": attempt_num,
                                        "success": False,
                                        "output": f"[Setup Pre-flight: {cmd_to_run}]\n{current_error}",
                                    }
                                )
                                ui.render_failure_panel(current_error, 1)
                                tester_review = _run_rca_and_archive(
                                    tester=tester,
                                    task_description=f"Analyze pre-flight pyproject.toml failure before `{cmd_to_run}` and perform Root Cause Analysis (RCA).",
                                    target_file="pyproject.toml",
                                    error_log=current_error,
                                    error_title=f"Setup Failure ({cmd_to_run}) in {project_folder_name}",
                                )
                                break

                        coder_logger.tool_call("Setup Command", cmd_to_run)
                        setup_res = terminal.run(cmd_to_run, cwd=str(project_workspace))
                        if not setup_res.success:
                            coder_logger.error(f"Setup command failed: {cmd_to_run}")
                            current_error = setup_res.stderr or setup_res.stdout or f"Command exited with code {setup_res.returncode}"
                            plan_execution_failed = True
                            execution_history.append(
                                {
                                    "attempt": attempt_num,
                                    "success": False,
                                    "output": f"[Setup Command: {cmd_to_run}]\n{current_error}",
                                }
                            )
                            ui.render_failure_panel(current_error, setup_res.returncode)
                            tester_review = _run_rca_and_archive(
                                tester=tester,
                                task_description=f"Analyze setup command failure ({cmd_to_run}) and perform Root Cause Analysis (RCA).",
                                target_file="pyproject.toml",
                                error_log=current_error,
                                error_title=f"Setup Failure ({cmd_to_run}) in {project_folder_name}",
                            )
                            break

            if plan_execution_failed:
                retry_count += 1
                should_break, formatted_instruction, retry_count = (
                    handle_max_retries_exhausted(
                        retry_count, execution_history, formatted_instruction, obsidian
                    )
                )
                if should_break:
                    break
                continue

        except Exception as api_err:
            sys_logger.error(
                f"Critical exception during agent execution phase: {api_err!s}"
            )
            obsidian.append_log(
                task_description=task_instruction,
                status="FAILED_API_ERROR",
                execution_history=[{"attempt": attempt_num, "output": str(api_err)}],
            )
            break

        # Step E: Main Terminal Verification Execution
        if not test_command or execution_mode == "none":
            sys_logger.stop_action("Smart Workflow completed successfully!")
            obsidian.append_log(
                task_description=task_instruction,
                status="SUCCESS",
                execution_history=execution_history,
            )
            break

        exec_result = terminal.run(test_command, cwd=str(project_workspace))

        output_text = (
            exec_result.stderr if not exec_result.success else exec_result.stdout
        )
        execution_history.append(
            {
                "attempt": attempt_num,
                "success": exec_result.success,
                "output": output_text,
            }
        )

        if exec_result.success:
            ui.render_success_panel(
                exec_result.stdout[:600]
                if exec_result.stdout
                else "All verification steps passed cleanly!"
            )
            obsidian.append_log(
                task_description=task_instruction,
                status="SUCCESS",
                execution_history=execution_history,
            )
            sys_logger.stop_action("Smart Workflow completed successfully!")
            break

        # Verification Failed: Initiate Self-Correction Loop
        retry_count += 1
        current_error = exec_result.stderr or exec_result.stdout or f"Command exited with code {exec_result.returncode}"

        ui.render_failure_panel(current_error, exec_result.returncode)
        tester_review = _run_rca_and_archive(
            tester=tester,
            task_description=f"Analyze failure in {target_file} and perform Root Cause Analysis (RCA).",
            target_file=target_file,
            error_log=current_error,
            error_title=f"Execution Failure in {target_file} ({project_folder_name})",
        )

        should_break, formatted_instruction, retry_count = (
            handle_max_retries_exhausted(
                retry_count, execution_history, formatted_instruction, obsidian
            )
        )
        if should_break:
            break


def main() -> None:
    """Run CLI workflow with graceful KeyboardInterrupt handling."""
    try:
        run_agent_workflow()
    except KeyboardInterrupt:
        ui.console.print()
        sys_logger.warning("Workflow cancelled by user. Terminating process.")
        sys.exit(0)
