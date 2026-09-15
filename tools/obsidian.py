import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


class ObsidianAdapter:
    """Safe adapter for reading rules and writing comprehensive logs in an existing PARA Obsidian Vault."""

    def __init__(self, vault_path: str):
        """
        :param vault_path: Absolute path to your Obsidian Vault.
        """
        self.vault_path = Path(vault_path)

    def read_rules(
        self,
        task_instruction: str = "",
        relative_rules_folder: str = "30_Resources/AgentRules",
    ) -> str:
        """
        Read core engineering rules (e.g., general, KISS, SOLID) unconditionally,
        and append domain-specific rules (e.g., HTTP, Security, Auth) only if matching
        keywords are found in the task instruction.

        :param task_instruction: User input task string used for keyword detection.
        :param relative_rules_folder: Path inside Vault where rules are stored.
        :return: Concatenated Markdown string of relevant rules.
        """
        target_dir = self.vault_path / relative_rules_folder

        # Create default rules structure if directory does not exist
        if not target_dir.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            default_rule = target_dir / "general_rules.md"
            default_rule.write_text(
                "# Core Principles\n- Keep It Simple, Stupid (KISS).\n- Don't Repeat Yourself (DRY).\n- Ensure proper error handling.\n",
                encoding="utf-8",
            )

        # Always read Core rules 
        core_rule_files = {
            "general_rules.md",
            "python_style.md",
        }

        # Keyword mapping for conditional domain-specific rule loading
        domain_keywords = {
            "http_requests.md": ["http", "api", "request", "rest", "endpoint", "fetch", "axios"],
            "security_auth.md": ["auth", "security", "token", "jwt", "login", "password", "oauth"],
            "database.md": ["db", "database", "sql", "sqlite", "orm", "query", "postgres"],
        }

        task_lower = task_instruction.lower()
        rule_texts = []

        for md_file in target_dir.glob("*.md"):
            filename = md_file.name.lower()

            # 1. Determine if this file is a domain-specific rule
            matched_domain_file = None
            for domain_file in domain_keywords:
                if domain_file in filename:
                    matched_domain_file = domain_file
                    break

            # 2. Filtering logic
            should_include = False

            # Always read core rules
            if filename in [f.lower() for f in core_rule_files]:
                should_include = True
                
            else:
                # Domain rules are included only if task contains relevant keywords
                keywords = domain_keywords[matched_domain_file]
                if any(kw in task_lower for kw in keywords):
                    should_include = True

            # 3. Read and append allowed rule files
            if should_include:
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