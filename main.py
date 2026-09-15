import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax

from config import Config
from agent import CodingAgent
from tools.obsidian import ObsidianAdapter
from tools.terminal import TerminalRunner

BASE_PROJECTS_DIR = "C:/Users/shoic/projects_with_agent"

# Reset rich console
console = Console()


def print_banner():
    """Display startup header panel."""
    banner_text = (
        f"[bold cyan]📁 Base Workspace Root:[/bold cyan] [dim]{BASE_PROJECTS_DIR}[/dim]\n"
        f"[bold cyan]📓 Obsidian Vault:[/bold cyan] [dim]{Config.OBSIDIAN_VAULT_PATH}[/dim]\n"
        f"[bold cyan]🔄 Max Retries per Attempt:[/bold cyan] [dim]{Config.MAX_RETRIES}[/dim]"
    )
    console.print(
        Panel(
            banner_text,
            title="[bold green]🚀 Autonomous Coding Agent[/bold green]",
            subtitle="[dim]Function Calling & Multi-Provider Fallback[/dim]",
            border_style="green",
            expand=False,
        )
    )


def run_agent_workflow():
    """Execute autonomous coding loop with isolated project directory generation."""
    print_banner()

    try:
        temp_agent = CodingAgent(workspace_root=BASE_PROJECTS_DIR)
        obsidian = ObsidianAdapter(str(Config.OBSIDIAN_VAULT_PATH))
    except Exception as e:
        console.print(f"[bold red]❌ Failed to initialize tools:[/bold red] {e}")
        sys.exit(1)

    # 1. User inputs task instruction
    console.print("\n[bold bold]💡 What do you want to build/fix?[/bold bold]")
    task_instruction = Prompt.ask("[bold yellow]Task[/bold yellow]").strip()

    if not task_instruction:
        console.print("[bold red]⚠️ Task instruction cannot be empty. Exiting.[/bold red]")
        return

    # 2. Automatically infer dedicated project folder name
    with console.status("[bold cyan]Inferring dedicated project folder name...", spinner="dots"):
        project_folder_name = temp_agent.suggest_project_folder(task_instruction)

    # Create absolute path for isolated project workspace
    project_workspace = Path(BASE_PROJECTS_DIR) / project_folder_name
    project_workspace.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel(
            f"[bold green]Directory Name:[/bold green] [cyan]{project_folder_name}[/cyan]\n"
            f"[bold green]Full Path:[/bold green] [dim]{project_workspace}[/dim]",
            title="[bold blue]📁 Dedicated Project Workspace Created[/bold blue]",
            border_style="blue",
            expand=False,
        )
    )

    # 3. Re-initialize CodingAgent targeting the isolated project workspace
    agent = CodingAgent(workspace_root=str(project_workspace))
    terminal = TerminalRunner(default_timeout=30)

    # 4. Infer main target script
    with console.status("[bold cyan]Inferring main target script...", spinner="dots"):
        target_file = agent.suggest_target_file(task_instruction)

    target_file_path = project_workspace / target_file

    # Append non-interactive constraint to task instruction
    task_instruction += (
        "\n\nIMPORTANT: Do NOT use interactive `input()`. "
        "Write unit tests or execution examples directly inside `if __name__ == '__main__':`."
    )

    # Determine execution test command
    if target_file.endswith(".py"):
        test_command = f'python "{target_file_path}"'
    else:
        test_command = "python -m pytest"

    console.print(f"[bold cyan]📄 Target Script:[/bold cyan] [green]{target_file}[/green]")
    console.print(f"[bold cyan]⚡ Auto Test Command:[/bold cyan] [yellow]{test_command}[/yellow]\n")

    # Read Obsidian rules
    with console.status("[bold cyan]Reading development rules from Obsidian...", spinner="dots"):
        obsidian_rules = obsidian.read_rules()

    execution_history = []
    retry_count = 0
    current_error = None

    console.rule(f"[bold magenta]Executing Task: {project_folder_name}[/bold magenta]")

    # 5. Autonomous Loop (Self-Correction & Fallback Execution)
    while True:
        attempt_num = retry_count + 1
        console.print(f"\n[bold cyan]🤖 [Agent Execution] Attempt {attempt_num}/{Config.MAX_RETRIES}...[/bold cyan]")

        try:
            with console.status("[bold cyan]Generating & applying solution via Tool Calls...", spinner="dots"):
                agent_response = agent.generate_code_solution(
                    task_instruction=task_instruction,
                    target_file_path=target_file,
                    obsidian_rules=obsidian_rules,
                    error_context=current_error,
                )
            console.print("[dim green]📝 Agent Tool Calls & Solution Generation Completed[/dim green]")

        except Exception as api_err:
            console.print(f"[bold red]❌ Error during agent execution:[/bold red] {api_err}")
            obsidian.append_log(
                task_description=task_instruction,
                status="FAILED",
                execution_history=[{
                    "attempt": attempt_num,
                    "success": False,
                    "output": f"API Exception: {str(api_err)}"
                }],
            )
            break

        # Run automated test in dedicated workspace
        console.print(f"[bold yellow]⚡ [Terminal] Running test command:[/bold yellow] [dim]{test_command}[/dim]")
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
                    exec_result.stdout[:500].strip() if exec_result.stdout else "No output",
                    title="[bold green]✅ [SUCCESS] Execution & Tests Passed Cleanly[/bold green]",
                    border_style="green",
                )
            )

            log_path = obsidian.append_log(
                task_description=task_instruction,
                status="SUCCESS",
                execution_history=execution_history,
            )
            console.print(f"[dim]📝 Log saved to Obsidian: {log_path}[/dim]")
            break

        else:
            retry_count += 1
            current_error = exec_result.stderr or exec_result.stdout

            console.print(
                Panel(
                    Syntax(current_error[:400].strip(), "text", line_numbers=False),
                    title=f"[bold red]❌ [EXECUTION FAILED] Exit Code: {exec_result.returncode}[/bold red]",
                    border_style="red",
                )
            )

            if retry_count >= Config.MAX_RETRIES:
                console.print(
                    Panel(
                        "[bold red]Reached maximum retry limit. Handing off to human.[/bold red]\n\n"
                        "[bold yellow]Options:[/bold yellow]\n"
                        "  • Type additional guidance/hints to retry\n"
                        "  • Press [bold white]ENTER[/bold white] or [bold white]'r'[/bold white] to retry as-is\n"
                        "  • Type [bold white]'q'[/bold white] to quit",
                        title="[bold yellow]⚠️ Human Intervention Required[/bold yellow]",
                        border_style="yellow",
                    )
                )

                user_feedback = Prompt.ask("[bold yellow]Your choice / guidance[/bold yellow]", default="r").strip()

                if user_feedback.lower() == "q":
                    console.print("[bold red]👋 Task terminated by user.[/bold red]")
                    break

                if user_feedback and user_feedback.lower() != "r":
                    task_instruction += f"\n\n[Human Guidance]: {user_feedback}"

                retry_count = 0
                current_error = None


if __name__ == "__main__":
    try:
        run_agent_workflow()
    except KeyboardInterrupt:
        console.print("\n[bold red]👋 Operation cancelled. Exiting.[/bold red]")
        sys.exit(0)put 