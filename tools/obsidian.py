import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


class ObsidianAdapter:
    """Safe adapter for reading rules, searching knowledge notes, writing logs, 
    and saving troubleshooting notes in an existing PARA Obsidian Vault.
    """

    def __init__(self, vault_path: str):
        """Initialize the Obsidian adapter with the vault base path.

        :param vault_path: Absolute path to your Obsidian Vault.
        """
        self.vault_path = Path(vault_path)

        # Core rule filenames (stored in lowercase for fast matching)
        self.core_rule_files = {
            "general_rules.md",
            "python_style.md",
        }

        # Keyword mapping for conditional domain-specific rule loading
        self.domain_keywords = {
            "http_requests.md": ["http", "api", "request", "rest", "endpoint", "fetch", "axios"],
            "security_auth.md": ["auth", "security", "token", "jwt", "login", "password", "oauth"],
            "database.md": ["db", "database", "sql", "sqlite", "orm", "query", "postgres"],
        }

    def search(self, query: str, max_results: int = 5) -> str:
        """Search across all Markdown notes in the Obsidian Vault for matching keywords or phrases.

        :param query: Keyword or space-separated search query.
        :param max_results: Maximum number of note snippets to return.
        :return: Formatted string containing matching file snippets.
        """
        if not self.vault_path.exists():
            return "Obsidian Vault directory does not exist."

        keywords = [k.lower() for k in query.split() if k.strip()]
        if not keywords:
            return "Empty search query provided."

        matches = []
        for md_file in self.vault_path.rglob("*.md"):
            # Ignore log files or hidden directories
            if ".obsidian" in md_file.parts or "50_Logs" in md_file.parts:
                continue

            try:
                content = md_file.read_text(encoding="utf-8")
                content_lower = content.lower()

                # Check if all or any keywords match
                if any(kw in content_lower for kw in keywords):
                    rel_path = md_file.relative_to(self.vault_path)
                    # Extract concise snippet
                    snippet_lines = [line for line in content.splitlines() if any(kw in line.lower() for kw in keywords)]
                    snippet = "\n".join(snippet_lines[:5])
                    matches.append(f"--- Note: [[{rel_path}]] ---\n{snippet}")

                    if len(matches) >= max_results:
                        break
            except Exception as e:
                continue

        if not matches:
            return f"No relevant notes or guidelines found in Obsidian Vault for query: '{query}'."

        return "\n\n".join(matches)

    def read_rules(
        self,
        task_instruction: str = "",
        relative_rules_folder: str = "30_Resources/AgentRules",
    ) -> str:
        """Read core engineering rules unconditionally, and append domain-specific rules
        only if matching keywords are found in the task instruction.
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

        task_lower = task_instruction.lower()
        rule_texts = []

        for md_file in target_dir.glob("*.md"):
            filename = md_file.name.lower()
            should_include = False

            # 1. Check if it is a core rule
            if filename in self.core_rule_files:
                should_include = True
            else:
                # 2. Check if it matches any registered domain rule pattern
                for domain_file, keywords in self.domain_keywords.items():
                    if domain_file in filename:
                        if any(kw in task_lower for kw in keywords):
                            should_include = True
                        break

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
        """Append detailed execution logs including intermediate attempt errors to Obsidian."""
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
            log_entry.append(f"<details>\n<summary>Attempt {attempt_num}: {icon}</summary>\n")
            log_entry.append(f"```text\n{output if output else 'No output returned.'}\n```")
            log_entry.append("\n</details>\n")

        log_entry.append("\n---\n")

        if not log_file_path.exists():
            header = f"# Agent Execution Logs ({now.strftime('%Y-%m')})\n\n"
            log_file_path.write_text(header, encoding="utf-8")

        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write("\n".join(log_entry) + "\n")

        return log_file_path

    def save_troubleshooting_note(
        self,
        title: str,
        content: str,
        relative_folder: str = "30_Resources/Troubleshooting",
    ) -> Path:
        """Saves a troubleshooting markdown note into the designated Vault directory.

        :param title: Clean title of the note for file naming.
        :param content: Full Markdown content with YAML frontmatter.
        :param relative_folder: Relative path inside Obsidian Vault.
        :return: Absolute Path object of the saved file.
        """
        target_dir = self.vault_path / relative_folder
        target_dir.mkdir(parents=True, exist_ok=True)

        date_prefix = datetime.now().strftime("%Y-%m-%d")
        
        # Clean title for safe filesystem naming
        clean_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")
        clean_title = re.sub(r"_+", "_", clean_title)
        filename = f"{date_prefix}_{clean_title}.md"

        # Clean content from surrounding markdown code blocks if agent wrapped entire note
        cleaned_content = content.strip()
        match = re.search(r"^```(?:markdown)?\s*([\s\S]*?)\s*```$", cleaned_content, re.IGNORECASE)
        if match:
            cleaned_content = match.group(1).strip()

        file_path = target_dir / filename
        file_path.write_text(cleaned_content, encoding="utf-8")
        return file_path