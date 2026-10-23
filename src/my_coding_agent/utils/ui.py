"""UI Module providing modern Rich terminal rendering, layout formatting, and agent logging."""

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.status import Status
from rich.syntax import Syntax
from rich.text import Text

# Global Rich console instance for system UI rendering
console = Console()

# Visual style configurations per agent role (Distinct colors assigned)
AGENT_STYLES: Dict[str, Dict[str, str]] = {
    "Planner": {"color": "cyan", "icon": "📌", "label": "PLANNER"},
    "Coder": {"color": "green", "icon": "💻", "label": "CODER"},
    "Tester": {"color": "yellow", "icon": "🧪", "label": "TESTER"},
    "System": {"color": "magenta", "icon": "🤖", "label": "SYSTEM"},
}


def sanitize_text(text: Any) -> str:
    """Sanitize raw text by replacing newlines with single spaces to ensure clean single-line terminal logs."""
    if text is None:
        return ""
    clean_str = str(text).replace("\r\n", " ").replace("\n", " ").strip()
    return " ".join(clean_str.split())


class AgentLogger:
    """Agent Logger encapsulating Rich-styled UI rendering and standard Python logging delegation."""

    def __init__(self, agent_name: str) -> None:
        """Initialize AgentLogger with visual style configurations and standard logger instance."""
        self.agent_name = agent_name
        style_info = AGENT_STYLES.get(
            agent_name, {"color": "blue", "icon": "📌", "label": agent_name.upper()}
        )
        self.color = style_info["color"]
        self.icon = style_info["icon"]
        self.label = style_info["label"]
        self._active_status: Optional[Status] = None
        self._last_status_text: Optional[str] = None
        self.py_logger = logging.getLogger(agent_name)

    def _format_status_text(self, action_description: str) -> Text:
        """Construct a styled Rich Text instance for active spinner descriptions."""
        clean_desc = sanitize_text(action_description)
        status_text = Text()
        status_text.append(f"[{self.label}] ", style=f"bold {self.color}")
        status_text.append(clean_desc, style="bold white")
        return status_text

    def _pause_active_status(self) -> Optional[str]:
        """Temporarily pause active spinner before printing log lines to prevent display artifacts."""
        if self._active_status is not None:
            saved_text = self._last_status_text
            try:
                self._active_status.stop()
            except Exception:
                pass
            self._active_status = None
            return saved_text
        return None

    def _resume_active_status(self, saved_text: Optional[str]) -> None:
        """Resume active spinner seamlessly after log lines have been printed."""
        if saved_text:
            self._active_status = console.status(
                self._format_status_text(saved_text), spinner="dots"
            )
            self._last_status_text = saved_text
            self._active_status.start()

    def _stop_active_status(self) -> None:
        """Safely stop and clear any active terminal spinner."""
        if self._active_status is not None:
            try:
                self._active_status.stop()
            except Exception:
                pass
            self._active_status = None
            self._last_status_text = None

    @contextmanager
    def status(self, action_description: str) -> Generator[None, None, None]:
        """Context manager to auto-manage animated spinner lifecycle during long operations or LLM calls."""
        self.start_action(action_description)
        try:
            yield
        finally:
            self._stop_active_status()

    def start_action(self, action_description: str) -> None:
        """Start an animated spinner or smoothly update description if already active."""
        clean_desc = sanitize_text(action_description)
        self._last_status_text = clean_desc
        if self._active_status is not None:
            self._active_status.update(status=self._format_status_text(clean_desc))
        else:
            self._active_status = console.status(
                self._format_status_text(clean_desc), spinner="dots"
            )
            self._active_status.start()
        self.py_logger.info(f"Started: {clean_desc}")

    def update_status(self, action_description: str) -> None:
        """Update the active spinner description dynamically without disrupting the animation."""
        clean_desc = sanitize_text(action_description)
        self._last_status_text = clean_desc
        if self._active_status is not None:
            self._active_status.update(status=self._format_status_text(clean_desc))
        else:
            self.start_action(clean_desc)
        self.py_logger.info(f"Updated status: {clean_desc}")

    def stop_action(self, completion_message: Optional[str] = None) -> None:
        """Safely stop active spinner and render a completion checkmark status line."""
        self._stop_active_status()

        if completion_message:
            clean_msg = sanitize_text(completion_message)
            escaped_msg = escape(clean_msg)
            console.print(
                f"[{self.color}]✔[/] [bold {self.color}][{self.label}][/] {escaped_msg}"
            )
            self.py_logger.info(f"Completed: {clean_msg}")

    def info(self, message: str) -> None:
        """Print standard informational logs safely formatted with agent identity styles."""
        saved_status = self._pause_active_status()

        clean_msg = sanitize_text(message)
        escaped_msg = escape(clean_msg)
        console.print(
            f"[{self.color}]{self.icon}[/] [bold {self.color}][{self.label}][/] {escaped_msg}"
        )
        self.py_logger.info(clean_msg)

        self._resume_active_status(saved_status)

    def log_llm_request(self, model_name: str, provider: str = "openrouter") -> None:
        """Log LiteLLM request without introducing leading or trailing blank line artifacts."""
        saved_status = self._pause_active_status()

        escaped_model = escape(sanitize_text(model_name))
        escaped_provider = escape(sanitize_text(provider))

        console.print(
            f"   [dim {self.color}]└─ 🌐 LiteLLM Request:[/"
            f" [cyan]{escaped_model}[/] [dim]via ({escaped_provider})[/]"
        )

        self._resume_active_status(saved_status)

    def key_value(self, key: str, value: Any) -> None:
        """Print a key-value status pair with unified syntax highlighting."""
        saved_status = self._pause_active_status()

        clean_key = sanitize_text(key)
        clean_val = sanitize_text(value)
        escaped_key = escape(clean_key)
        escaped_val = escape(clean_val)
        console.print(
            f"[{self.color}]{self.icon}[/] [bold {self.color}][{self.label}][/] "
            f"[bold cyan]{escaped_key}:[/bold cyan] [bold yellow]{escaped_val}[/bold yellow]"
        )
        self.py_logger.info(f"{clean_key}: {clean_val}")

        self._resume_active_status(saved_status)

    def tool_call(self, tool_name: str, args_summary: str = "") -> None:
        """Log tool invocation events with dynamic argument type labels (path, dir, query)."""
        saved_status = self._pause_active_status()

        clean_tool = sanitize_text(tool_name)
        clean_args = sanitize_text(args_summary)

        if len(clean_args) > 120:
            clean_args = clean_args[:117] + "..."

        if clean_tool in ["write_file", "read_file", "delete_file"]:
            param_label = "path"
        elif clean_tool == "make_directory":
            param_label = "dir"
        else:
            param_label = "query"

        summary_str = f' ({param_label}="{clean_args}")' if clean_args else ""

        escaped_tool = escape(clean_tool)
        escaped_summary = escape(summary_str)

        console.print(
            f"   [dim {self.color}]└─ 🛠 Executing Tool:["
            f" [bold cyan]{escaped_tool}[/]"
            f"[dim]{escaped_summary}[/]]"
        )
        self.py_logger.info(f"Tool executed: {clean_tool} {summary_str}".strip())

        self._resume_active_status(saved_status)

    def tool_result(self, detail: Union[str, List[str]], max_display: int = 3) -> None:
        """Log structured tool output details or file paths in a formatted tree view."""
        saved_status = self._pause_active_status()

        if isinstance(detail, str):
            items = [sanitize_text(item) for item in detail.split(",") if sanitize_text(item)]
        else:
            items = [sanitize_text(item) for item in detail if sanitize_text(item)]

        if not items:
            self._resume_active_status(saved_status)
            return

        total = len(items)
        if total == 1:
            escaped_item = escape(items[0])
            console.print(
                f"      [dim {self.color}]└─ 📄 Read:[/] [dim cyan]{escaped_item}[/dim cyan]"
            )
        else:
            console.print(
                f"      [dim {self.color}]└─ 📄 Read ({total} files):[/]"
            )
            displayed_items = items[:max_display]
            for idx, item in enumerate(displayed_items):
                is_last_item = (idx == len(displayed_items) - 1) and (total <= max_display)
                branch = "└─" if is_last_item else "├─"
                escaped_item = escape(item)
                console.print(
                    f"         [dim {self.color}]{branch}[/] [dim cyan]{escaped_item}[/dim cyan]"
                )

            if total > max_display:
                remaining = total - max_display
                console.print(
                    f"         [dim {self.color}]└─[/] [dim yellow]... and {remaining} more[/dim yellow]"
                )

        self.py_logger.info(f"Tool result details: {', '.join(items)}")

        self._resume_active_status(saved_status)

    def warning(self, message: str) -> None:
        """Print warning messages with highlighted warning styling."""
        saved_status = self._pause_active_status()

        clean_msg = sanitize_text(message)
        escaped_msg = escape(clean_msg)
        console.print(f"[bold yellow]⚠️ [{self.label}] Warning:[/] {escaped_msg}")
        self.py_logger.warning(clean_msg)

        self._resume_active_status(saved_status)

    def error(self, message: str) -> None:
        """Print error logs enclosed within a styled Rich Panel."""
        saved_status = self._pause_active_status()

        clean_msg = sanitize_text(message)
        escaped_msg = escape(clean_msg)
        console.print(
            Panel(
                f"[bold red]{escaped_msg}[/]",
                title=f"❌ [{self.label}] Error",
                border_style="red",
            )
        )
        self.py_logger.error(clean_msg)

        self._resume_active_status(saved_status)


def render_banner(workspace_root: Union[str, Path], obsidian_vault: Union[str, Path], max_retries: int) -> None:
    """Render the application startup dashboard panel."""
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
    """Render a horizontal rule separator for primary execution workflow phases."""
    clean_title = sanitize_text(phase_title)
    console.rule(f"[bold magenta]{escape(clean_title)}[/bold magenta]")


def render_success_panel(message: str) -> None:
    """Render an execution success outcome panel."""
    console.print(
        Panel(
            escape(message.strip()),
            title="[bold green]✅ [SUCCESS] Verification Passed Cleanly[/bold green]",
            border_style="green",
        )
    )


def render_failure_panel(error_log: str, return_code: int) -> None:
    """Render a failure execution panel with syntax highlighting for terminal logs."""
    console.print(
        Panel(
            Syntax(error_log[:500].strip(), "text", line_numbers=False),
            title=f"[bold red]❌ [EXECUTION FAILED] Exit Code: {return_code}[/bold red]",
            border_style="red",
        )
    )


def render_rca_report_panel(report_text: str) -> None:
    """Render the Root Cause Analysis (RCA) review panel generated by the Tester Agent."""
    console.print(
        Panel(
            escape(report_text[:600].strip()),
            title="[bold magenta]🔍 [TESTER REVIEW / RCA REPORT][/bold magenta]",
            border_style="magenta",
        )
    )


def render_human_intervention_panel() -> None:
    """Render the interactive human-in-the-loop fallback panel."""
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