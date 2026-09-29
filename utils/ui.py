"""UI Module handling all Rich rendering, layout formatting, and agent logging."""

import logging
from pathlib import Path
from typing import Optional, Union
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.status import Status
from rich.syntax import Syntax
from rich.text import Text

# Global Rich console instance
console = Console()

# Defined visual configurations per agent role
AGENT_STYLES = {
    "Planner": {"color": "cyan", "icon": "🧠", "label": "PLANNER"},
    "Coder": {"color": "green", "icon": "⚡", "label": "CODER"},
    "Tester": {"color": "yellow", "icon": "🔍", "label": "TESTER"},
    "System": {"color": "magenta", "icon": "🤖", "label": "SYSTEM"},
}


class AgentLogger:
    """Agent Logger encapsulating Rich style rendering and Python logging delegator."""

    def __init__(self, agent_name: str) -> None:
        """Initialize AgentLogger with styled colors and icons."""
        self.agent_name = agent_name
        style_info = AGENT_STYLES.get(
            agent_name, {"color": "blue", "icon": "📌", "label": agent_name.upper()}
        )
        self.color = style_info["color"]
        self.icon = style_info["icon"]
        self.label = style_info["label"]
        self._active_status: Optional[Status] = None
        self.py_logger = logging.getLogger(agent_name)

    def start_action(self, action_description: str) -> None:
        """Display animated terminal spinner for background tasks."""
        if self._active_status:
            self._active_status.stop()

        status_text = Text()
        status_text.append(f"[{self.label}] ", style=f"bold {self.color}")
        status_text.append(action_description, style="bold white")

        self._active_status = console.status(status_text, spinner="dots")
        self._active_status.start()
        self.py_logger.info(f"Started: {action_description}")

    def stop_action(self, completion_message: Optional[str] = None) -> None:
        """Stop active spinner and print completion status."""
        if self._active_status:
            self._active_status.stop()
            self._active_status = None

        if completion_message:
            escaped_msg = escape(completion_message)
            console.print(
                f"[{self.color}]✔[/] [bold {self.color}][{self.label}][/] {escaped_msg}"
            )
            self.py_logger.info(f"Completed: {completion_message}")

    def info(self, message: str) -> None:
        """Print plain text informational logs formatted safely with agent styles."""
        if self._active_status:
            self._active_status.stop()
            self._active_status = None

        escaped_msg = escape(message)
        console.print(
            f"[{self.color}]{self.icon}[/] [bold {self.color}][{self.label}][/] {escaped_msg}"
        )
        self.py_logger.info(message)

    def key_value(self, key: str, value: Any) -> None:
        """Print key-value system status pair with consistent highlighting."""
        if self._active_status:
            self._active_status.stop()
            self._active_status = None

        escaped_key = escape(key)
        escaped_val = escape(str(value))
        console.print(
            f"[{self.color}]{self.icon}[/] [bold {self.color}][{self.label}][/] "
            f"[bold cyan]{escaped_key}:[/bold cyan] [bold yellow]{escaped_val}[/bold yellow]"
        )
        self.py_logger.info(f"{key}: {value}")

    def tool_call(self, tool_name: str, args_summary: str = "") -> None:
        """Log tool execution events with target summaries."""
        summary_str = f"({args_summary})" if args_summary else ""
        escaped_tool = escape(tool_name)
        escaped_summary = escape(summary_str)

        console.print(
            f"   [dim {self.color}]└─ 🛠 Tool Executed:[/]"
            f" [bold cyan]{escaped_tool}[/]"
            f" [dim]{escaped_summary}[/]"
        )
        self.py_logger.info(f"Tool executed: {tool_name} {summary_str}".strip())

    def warning(self, message: str) -> None:
        """Print warning level messages."""
        escaped_msg = escape(message)
        console.print(f"[bold yellow]⚠️ [{self.label}] Warning:[/] {escaped_msg}")
        self.py_logger.warning(message)

    def error(self, message: str) -> None:
        """Print error logs wrapped inside styled Rich Panel."""
        escaped_msg = escape(message)
        console.print(
            Panel(
                f"[bold red]{escaped_msg}[/]",
                title=f"❌ [{self.label}] Error",
                border_style="red",
            )
        )
        self.py_logger.error(message)


# -------------------------------------------------------------------
# Standalone UI Rendering Functions (Decoupled from main.py)
# -------------------------------------------------------------------

def render_banner(workspace_root: Union[str, Path], obsidian_vault: Union[str, Path], max_retries: int) -> None:
    """Render startup dashboard panel."""
    banner_text = (
        f"[bold cyan]📁 Workspace Root:[/bold cyan] [dim]{escape(str(workspace_root))}[/dim]\n"
        f"[bold cyan]📓 Obsidian Vault:[/bold cyan] [dim]{escape(str(obsidian_vault))}[/dim]\n"
        f"[bold cyan]🔄 Max Retries per Phase:[/bold cyan] [dim]{max_retries}[/dim]"
    )
    console.print(
        Panel(
            banner_text,
            title="[bold green]🚀 Autonomous Multi-Agent Engine[/bold green]",
            subtitle="[dim]Closed-Loop Self-Correction & Knowledge Archiving System[/dim]",
            border_style="green",
            expand=False,
        )
    )


def render_phase_header(phase_title: str) -> None:
    """Render horizontal rule separator for major execution workflow phases."""
    console.rule(f"[bold magenta]{escape(phase_title)}[/bold magenta]")


def render_success_panel(message: str) -> None:
    """Render execution success outcome panel."""
    console.print(
        Panel(
            escape(message.strip()),
            title="[bold green]✅ [SUCCESS] Verification Passed Cleanly[/bold green]",
            border_style="green",
        )
    )


def render_failure_panel(error_log: str, return_code: int) -> None:
    """Render failure terminal log panel with syntax highlighting."""
    console.print(
        Panel(
            Syntax(error_log[:500].strip(), "text", line_numbers=False),
            title=f"[bold red]❌ [EXECUTION FAILED] Exit Code: {return_code}[/bold red]",
            border_style="red",
        )
    )


def render_rca_report_panel(report_text: str) -> None:
    """Render Tester Agent RCA Review panel."""
    console.print(
        Panel(
            escape(report_text[:600].strip()),
            title="[bold magenta]🔍 [TESTER REVIEW / RCA REPORT][/bold magenta]",
            border_style="magenta",
        )
    )


def render_human_intervention_panel() -> None:
    """Render human intervention menu option panel."""
    console.print(
        Panel(
            "[bold red]Maximum retry limit reached. Manual guidance required.[/bold red]\n\n"
            "[bold yellow]Options:[/bold yellow]\n"
            "  • Enter custom hints/feedback to guide next iteration\n"
            "  • Press [bold white]ENTER[/bold white] or [bold white]'r'[/bold white] to retry\n"
            "  • Enter [bold white]'q'[/bold white] to abort workflow",
            title="[bold yellow]⚠️ Human Intervention Required[/bold yellow]",
            border_style="yellow",
        )
    )