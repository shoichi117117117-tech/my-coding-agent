import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


class ObsidianAdapter:
    """Safe adapter for reading rules and writing comprehensive logs in an existing PARA Obsidian Vault."""

    def __init__(self, vault_path: str):
        """
        :param vault_path: Absolute path to your Obsidian Vault.
        """
        self.vault_path = Path(vault_path)

    def read_rules(self, relative_rules_folder: str = "30_Resources/AgentRules") -> str:
        """Read markdown rule files from the target PARA Vault directory."""
        target_dir = self.vault_path / relative_rules_folder

        if not target_dir.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            default_rule = target_dir / "general_rules.md"
            default_rule.write_text(
                "# General Development Rules\n- Write clean, documented code.\n- Ensure all errors are logged.\n",
                encoding="utf-8",
            )

        rule_texts = []
        for md_file in target_dir.glob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                rule_texts.append(f"--- Rule: {md_file.name} ---\n{content}")
            except Exception as e:
                rule_texts.append(f"--- Rule: {md_file.name} (Read Error: {e}) ---")

        return "\n\n".join(rule_texts) if rule_texts else "No specific rules found."

    def append_log(
        self,
        task_description: str,
        status: str,
        execution_history: List[Dict[str, Any]],
        relative_log_folder: str = "50_Logs/AgentLogs",
    ) -> Path:
        """
        Append detailed execution logs including intermediate attempt errors to Obsidian.

        :param task_description: Description of user task.
        :param status: Execution status ("SUCCESS" / "FAILED").
        :param execution_history: List of dicts containing attempt numbers and error details.
        :param relative_log_folder: Path inside Vault reserved for logs.
        :return: Path to updated log file.
        """
        target_dir = self.vault_path / relative_log_folder
        target_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now()
        log_file_name = f"agent_log_{now.strftime('%Y-%m')}.md"
        log_file_path = target_dir / log_file_name

        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

        log_entry = [
            f"### [{timestamp}] Task: {task_description.splitlines()[0]}",
            f"- **Final Status**: `{status}`",
            f"- **Total Attempts**: `{len(execution_history)}`",
            "\n#### 🔄 Execution History & Self-Correction Trail:\n",
        ]

        for item in execution_history:
            attempt_num = item.get("attempt", 1)
            is_pass = item.get("success", False)
            output = item.get("output", "").strip()

            icon = "✅ Pass" if is_pass else "❌ Error"
            log_entry.append(f"<details><summary>Attempt {attempt_num}: {icon}</summary>\n")
            log_entry.append(f"```text\n{output if output else 'No output returned.'}\n```")
            log_entry.append("\n</details>\n")

        log_entry.append("\n---\n")

        if not log_file_path.exists():
            header = f"# Agent Execution Logs ({now.strftime('%Y-%m')})\n\n"
            log_file_path.write_text(header, encoding="utf-8")

        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write("\n".join(log_entry) + "\n")

        return log_file_path