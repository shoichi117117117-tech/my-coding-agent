"""Main entry point for the autonomous multi-agent development engine."""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.prompt import Prompt

from agents.coder import CoderAgent
from agents.planner import PlannerAgent
from agents.tester import TesterAgent
from config import Config
from tools.obsidian import ObsidianAdapter
from tools.terminal import TerminalRunner
from utils.logger_config import setup_file_logging
from utils import ui

# Initialize file logging persistence at system startup
setup_file_logging()

# Global workspace path configuration with fallback default
BASE_PROJECTS_DIR = Path(getattr(Config, "BASE_PROJECTS_DIR", Path.home() / "projects_with_agent"))

# Unified agent loggers
sys_logger = ui.AgentLogger("System")
planner_logger = ui.AgentLogger("Planner")
coder_logger = ui.AgentLogger("Coder")
tester_logger = ui.AgentLogger("Tester")


def parse_plan_steps(plan_output: Any) -> List[Dict[str, Any]]:
    """Safely parse structured strategy plan steps from JSON string or dictionary."""
    if isinstance(plan_output, list):
        return plan_output
    if isinstance(plan_output, dict):
        return plan_output.get("steps", [])

    if isinstance(plan_output, str):
        cleaned = plan_output.strip()
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed.get("steps", [])
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    return []


def run_agent_workflow() -> None:
    """Execute modern autonomous coding workflow with closed-loop multi-agent orchestration."""
    ui.render_banner(BASE_PROJECTS_DIR, Config.OBSIDIAN_VAULT_PATH, Config.MAX_RETRIES)

    # Step 1: Initialize adapters and tools safely
    try:
        obsidian = ObsidianAdapter(str(Config.OBSIDIAN_VAULT_PATH))
        terminal = TerminalRunner(default_timeout=30)
    except Exception as e:
        sys_logger.error(f"Failed to initialize system adapters: {e}")
        sys.exit(1)

    # Step 2: Acquire task instruction from user
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
        workspace_root=str(BASE_PROJECTS_DIR),
        obsidian_vault_path=str(Config.OBSIDIAN_VAULT_PATH),
    )

    project_folder_name = planner.suggest_project_folder(task_instruction)

    project_workspace = BASE_PROJECTS_DIR / project_folder_name
    project_workspace.mkdir(parents=True, exist_ok=True)

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
    strategy = planner.analyze_verification_strategy(task_instruction)

    target_file = str(strategy.get("target_file") or "main.py")
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

    execution_history: List[Dict[str, Any]] = []
    retry_count = 0
    current_error: Optional[str] = None
    tester_review: Optional[str] = None

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

                full_test_path = project_workspace / test_file
                full_test_path.parent.mkdir(parents=True, exist_ok=True)
                full_test_path.write_text(test_code, encoding="utf-8")
                tester_logger.stop_action(f"Test suite stored at: {test_file}")

            # Step C: Coder Implements/Refactors Codebase
            coder_prompt = (
                f"Implement or update the codebase based on the execution plan below.\n"
                f"If specific configuration manifests (e.g., pyproject.toml, package.json) are required by the plan, "
                f"use `search_obsidian_vault` tool to verify correct schemas if needed.\n\n"
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

            # Step D: Sequential Plan Execution & Setup Commands
            plan_execution_failed = False
            for step in plan_steps:
                cmd_to_run = step.get("command_to_execute")
                if cmd_to_run and test_command and cmd_to_run.strip() != test_command.strip():
                    coder_logger.tool_call("Setup Command", cmd_to_run)
                    setup_res = terminal.run(cmd_to_run, cwd=str(project_workspace))
                    if not setup_res.success:
                        coder_logger.error(f"Setup command failed: {cmd_to_run}")
                        current_error = setup_res.stderr or setup_res.stdout
                        plan_execution_failed = True
                        break

            if plan_execution_failed:
                retry_count += 1
                continue

        except Exception as api_err:
            sys_logger.error(f"Critical exception during agent execution phase: {str(api_err)}")
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

        output_text = exec_result.stderr if not exec_result.success else exec_result.stdout
        execution_history.append({
            "attempt": attempt_num,
            "success": exec_result.success,
            "output": output_text,
        })

        if exec_result.success:
            ui.render_success_panel(
                exec_result.stdout[:600] if exec_result.stdout else "All verification steps passed cleanly!"
            )
            log_path = obsidian.append_log(
                task_description=task_instruction,
                status="SUCCESS",
                execution_history=execution_history,
            )
            sys_logger.stop_action("Smart Workflow completed successfully!")
            break

        # Verification Failed: Initiate Self-Correction Loop
        retry_count += 1
        current_error = exec_result.stderr or exec_result.stdout

        ui.render_failure_panel(current_error, exec_result.returncode)

        tester_response = tester.execute_task(
            task_description=f"Analyze failure in {target_file} and perform Root Cause Analysis (RCA).",
            target_file=target_file,
            execution_log=current_error,
            auto_fetch_vault_rules=False,
        )

        tester_review = getattr(tester_response, "content", str(tester_response))

        ui.render_rca_report_panel(tester_review)

        tester.generate_obsidian_troubleshooting_note(
            error_title=f"Execution Failure in {target_file} ({project_folder_name})",
            error_log=current_error,
            solution_summary=tester_review,
            save_to_vault=True,
            relative_folder="30_Resources/Troubleshooting",
        )

        if retry_count >= Config.MAX_RETRIES:
            ui.render_human_intervention_panel()

            user_feedback = Prompt.ask("[bold yellow]Decision[/bold yellow]", default="r").strip()

            if user_feedback.lower() == "q":
                sys_logger.warning("Workflow execution aborted by user.")
                obsidian.append_log(
                    task_description=task_instruction,
                    status="ABORTED_BY_USER",
                    execution_history=execution_history,
                )
                break

            if user_feedback and user_feedback.lower() != "r":
                formatted_instruction += f"\n\n[Human Operator Guidance]: {user_feedback}"

            retry_count = 0


if __name__ == "__main__":
    try:
        run_agent_workflow()
    except KeyboardInterrupt:
        ui.console.print()
        sys_logger.warning("Workflow cancelled by user. Terminating process.")
        sys.exit(0)