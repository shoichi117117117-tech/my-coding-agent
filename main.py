import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax

from config import Config
from agents.planner import PlannerAgent
from agents.coder import CoderAgent
from agents.tester import TesterAgent
from tools.obsidian import ObsidianAdapter
from tools.terminal import TerminalRunner

# Constants & Initializations
BASE_PROJECTS_DIR = Path(getattr(Config, "BASE_PROJECTS_DIR", "C:/Users/shoic/projects_with_agent"))
console = Console()


def print_banner() -> None:
    """Display startup header panel."""
    banner_text = (
        f"[bold cyan]📁 Base Workspace Root:[/bold cyan] [dim]{BASE_PROJECTS_DIR}[/dim]\n"
        f"[bold cyan]📓 Obsidian Vault:[/bold cyan] [dim]{Config.OBSIDIAN_VAULT_PATH}[/dim]\n"
        f"[bold cyan]🔄 Max Retries per Phase:[/bold cyan] [dim]{Config.MAX_RETRIES}[/dim]"
    )
    console.print(
        Panel(
            banner_text,
            title="[bold green]🚀 Modern Autonomous Multi-Agent Engine[/bold green]",
            subtitle="[dim]Closed-Loop Self-Correction & Knowledge Archiving System[/dim]",
            border_style="green",
            expand=False,
        )
    )


def parse_plan_steps(plan_output: Any) -> List[Dict[str, Any]]:
    """Parse JSON strategy plan steps safely."""
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
    """Execute modern autonomous coding workflow with dynamic verification modes."""
    print_banner()

    # 1. Initialize tools and adapters
    try:
        obsidian = ObsidianAdapter(str(Config.OBSIDIAN_VAULT_PATH))
        terminal = TerminalRunner(default_timeout=30)
    except Exception as e:
        console.print(f"[bold red]❌ Failed to initialize tools:[/bold red] {e}")
        sys.exit(1)

    # 2. Acquire task instruction from user
    console.print("\n[bold]💡 What do you want to build or fix?[/bold]")
    task_instruction = Prompt.ask("[bold yellow]Task Prompt[/bold yellow]").strip()

    if not task_instruction:
        console.print("[bold red]⚠️ Task instruction cannot be empty. Exiting.[/bold red]")
        return

    # ---------------------------------------------------------
    # Phase 1: Workspace Isolation
    # ---------------------------------------------------------
    planner = PlannerAgent(
        workspace_root=str(BASE_PROJECTS_DIR),
        obsidian_vault_path=str(Config.OBSIDIAN_VAULT_PATH),
    )

    # Pass the pure user instruction to suggest_project_folder
    with console.status("[bold cyan]Phase 1/5: Planner inferring project environment...", spinner="dots"):
        project_folder_name = planner.suggest_project_folder(task_instruction)

    project_workspace = BASE_PROJECTS_DIR / project_folder_name
    project_workspace.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel(
            f"[bold green]Directory Name:[/bold green] [cyan]{project_folder_name}[/cyan]\n"
            f"[bold green]Full Workspace Path:[/bold green] [dim]{project_workspace}[/dim]",
            title="[bold blue]📁 Phase 1 Completed: Workspace Isolated[/bold blue]",
            border_style="blue",
            expand=False,
        )
    )

    # Non-interactive execution constraint
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
    with console.status("[bold cyan]Phase 2/5: Reading Obsidian rules & analyzing strategy...", spinner="dots"):
        obsidian_rules = obsidian.read_rules(task_instruction=task_instruction)
        strategy = planner.analyze_verification_strategy(task_instruction)

        target_file = strategy.get("target_file", "main.py")
        test_file = strategy.get("test_file")
        test_command = strategy.get("test_command")
        execution_mode = strategy.get("mode", "direct_execution")

    console.print(f"[bold cyan]📄 Target File:[/bold cyan]     [green]{target_file}[/green]")
    console.print(f"[bold cyan]🎯 Execution Mode:[/bold cyan]  [yellow]{execution_mode}[/yellow]")
    if test_file:
        console.print(f"[bold cyan]🧪 Test File:[/bold cyan]       [green]{test_file}[/green]")
    if test_command:
        console.print(f"[bold cyan]⚡ Verification Cmd:[/bold cyan] [yellow]{test_command}[/yellow]\n")

    # ---------------------------------------------------------
    # Initialize Core Agents
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

    console.rule(f"[bold magenta]Executing Smart Workflow: {project_folder_name}[/bold magenta]")

    # ---------------------------------------------------------
    # Phase 3~5: Closed-Loop ReAct / Self-Correction Cycle
    # ---------------------------------------------------------
    while True:
        attempt_num = retry_count + 1
        console.print(f"\n[bold cyan]🤖 [Iteration {attempt_num}/{Config.MAX_RETRIES}][/bold cyan]")

        try:
            # -------------------------------------------------
            # Step A: Planner Generates/Updates Structured Plan
            # -------------------------------------------------
            with console.status("[bold cyan]PlannerAgent: Processing feedback & creating plan...", spinner="dots"):
                planner_error_context = None
                if current_error:
                    planner_error_context = (
                        f"【Raw Execution Log】\n{current_error}\n\n"
                        f"【Tester Agent RCA Review & Guidance】\n{tester_review if tester_review else 'N/A'}"
                    )

                raw_plan_json = planner.create_plan(
                    task_instruction=formatted_instruction,
                    target_file=target_file,
                    domain_rules=obsidian_rules,
                    error_context=planner_error_context,
                    attempt=attempt_num,
                    auto_fetch_vault_rules=True,
                )
                plan_steps = parse_plan_steps(raw_plan_json)

            console.print(f"[dim green]🧠 Strategy plan updated by Planner ({len(plan_steps)} steps).[/dim green]")

            # -------------------------------------------------
            # Step B: Tester Generates Test Cases (Only if mode is unit_test)
            # -------------------------------------------------
            if execution_mode == "unit_test" and test_file:
                with console.status("[bold cyan]TesterAgent: Generating test suite with Obsidian context...", spinner="dots"):
                    existing_code = ""
                    source_path = project_workspace / target_file
                    if source_path.exists():
                        existing_code = source_path.read_text(encoding="utf-8")

                    test_code = tester.generate_test_code(
                        target_file=target_file,
                        code_content=existing_code,
                        test_requirements=f"Plan:\n{raw_plan_json}\n\nTask:\n{formatted_instruction}",
                        auto_fetch_vault_rules=True,
                    )

                    full_test_path = project_workspace / test_file
                    full_test_path.parent.mkdir(parents=True, exist_ok=True)
                    full_test_path.write_text(test_code, encoding="utf-8")

                console.print(f"[dim green]🧪 Test suite generated and written to: {test_file}[/dim green]")
            else:
                console.print("[dim yellow]ℹ️ Skipping unit test generation (Non-unit-test execution mode).[/dim yellow]")

            # -------------------------------------------------
            # Step C: Coder Implements/Refactors Codebase
            # -------------------------------------------------
            with console.status("[bold cyan]CoderAgent: Generating / fixing source code...", spinner="dots"):
                coder_prompt = (
                    f"Implement or update the codebase based on the execution plan below.\n"
                    f"If specific configuration manifests (e.g. pyproject.toml, package.json) are required by the plan, "
                    f"use `search_obsidian_vault` tool to verify correct schemas if needed.\n\n"
                    f"Execution Plan:\n{raw_plan_json}\n\n"
                    f"User Instruction:\n{formatted_instruction}"
                )
                coder.execute_task(
                    task_description=coder_prompt,
                    rules=obsidian_rules,
                    error_context=planner_error_context,
                    auto_fetch_vault_rules=True,
                )
            console.print(f"[dim green]📝 Source code updated for target: {target_file}[/dim green]")

            # -------------------------------------------------
            # Step D: Sequential Plan Execution & Setup Syncing
            # -------------------------------------------------
            plan_execution_failed = False
            for step in plan_steps:
                cmd_to_run = step.get("command_to_execute")
                if cmd_to_run and cmd_to_run.strip() != test_command.strip():
                    console.print(f"[bold yellow]⚙️ [Setup Command]:[/bold yellow] [dim]{cmd_to_run}[/dim]")
                    setup_res = terminal.run(cmd_to_run, cwd=str(project_workspace))
                    if not setup_res.success:
                        console.print(f"[bold red]❌ Setup command failed: {cmd_to_run}[/bold red]")
                        current_error = setup_res.stderr or setup_res.stdout
                        plan_execution_failed = True
                        break

            if plan_execution_failed:
                retry_count += 1
                continue

        except Exception as api_err:
            console.print(f"[bold red]❌ Critical API Error during Agent Execution Phase:[/bold red] {api_err}")
            obsidian.append_log(
                task_description=task_instruction,
                status="FAILED_API_ERROR",
                execution_history=[{"attempt": attempt_num, "output": str(api_err)}],
            )
            break

        # -------------------------------------------------
        # Step E: Main Terminal Verification Execution
        # -------------------------------------------------
        if not test_command or execution_mode == "none":
            console.print("[bold green]✅ No execution command required. Generation complete![/bold green]")
            obsidian.append_log(
                task_description=task_instruction,
                status="SUCCESS",
                execution_history=execution_history,
            )
            break

        console.print(f"[bold yellow]⚡ [Verification Execution]:[/bold yellow] [dim]{test_command}[/dim]")
        exec_result = terminal.run(test_command, cwd=str(project_workspace))

        output_text = exec_result.stderr if not exec_result.success else exec_result.stdout
        execution_history.append({
            "attempt": attempt_num,
            "success": exec_result.success,
            "output": output_text,
        })

        if exec_result.success:
            console.print(
                Panel(
                    exec_result.stdout[:600].strip() if exec_result.stdout else "All verification steps passed cleanly!",
                    title="[bold green]✅ [SUCCESS] Verification Passed Cleanly[/bold green]",
                    border_style="green",
                )
            )

            log_path = obsidian.append_log(
                task_description=task_instruction,
                status="SUCCESS",
                execution_history=execution_history,
            )
            console.print(f"[dim green]📝 Main execution log saved to Obsidian: {log_path}[/dim green]")
            break
        else:
            retry_count += 1
            current_error = exec_result.stderr or exec_result.stdout

            console.print(
                Panel(
                    Syntax(current_error[:500].strip(), "text", line_numbers=False),
                    title=f"[bold red]❌ [EXECUTION FAILED] Exit Code: {exec_result.returncode}[/bold red]",
                    border_style="red",
                )
            )

            with console.status("[bold magenta]TesterAgent: Performing RCA & preparing review for Planner...", spinner="dots"):
                tester_response = tester.execute_task(
                    task_description=f"Analyze failure in {target_file} and perform Root Cause Analysis (RCA).",
                    target_file=target_file,
                    execution_log=current_error,
                    auto_fetch_vault_rules=True,
                )

                tester_review = getattr(tester_response, "content", str(tester_response))

            console.print(
                Panel(
                    tester_review[:600].strip(),
                    title="[bold magenta]🔍 [TESTER REVIEW / RCA REPORT][/bold magenta]",
                    border_style="magenta",
                )
            )

            with console.status("[bold magenta]TesterAgent: Archiving troubleshooting note to Obsidian...", spinner="dots"):
                tester.generate_obsidian_troubleshooting_note(
                    error_title=f"Execution Failure in {target_file} ({project_folder_name})",
                    error_log=current_error,
                    solution_summary=tester_review,
                    save_to_vault=True,
                    relative_folder="30_Resources/Troubleshooting",
                )
            console.print("[dim magenta]📓 Troubleshooting document saved to Obsidian Vault.[/dim magenta]")

            if retry_count >= Config.MAX_RETRIES:
                console.print(
                    Panel(
                        "[bold red]Reached maximum retry limit. Handing off to human operator.[/bold red]\n\n"
                        "[bold yellow]Options:[/bold yellow]\n"
                        "  • Type specific hints/feedback to guide next retry attempt\n"
                        "  • Press [bold white]ENTER[/bold white] or [bold white]'r'[/bold white] to retry without new hints\n"
                        "  • Type [bold white]'q'[/bold white] to quit execution",
                        title="[bold yellow]⚠️ Human Intervention Required[/bold yellow]",
                        border_style="yellow",
                    )
                )

                user_feedback = Prompt.ask("[bold yellow]Your decision[/bold yellow]", default="r").strip()

                if user_feedback.lower() == "q":
                    console.print("[bold red]👋 Task execution terminated by user.[/bold red]")
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
        console.print("\n[bold red]👋 Operation cancelled by user. Exiting gracefully.[/bold red]")
        sys.exit(0)