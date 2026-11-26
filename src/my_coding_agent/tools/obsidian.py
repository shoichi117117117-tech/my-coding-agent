import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# --- Tool / Obsidian Adapter Implementation ---
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

    def search_with_sources(
        self, query: Union[str, List[str]], max_results: int = 5
    ) -> Tuple[str, List[str]]:
        """Search across all Markdown notes in Obsidian Vault with relevance scoring.

        :param query: Raw search query string or pre-extracted clean keyword list.
        :param max_results: Maximum number of top relevant note snippets to return.
        :return: Tuple containing (formatted_result_text, list_of_relative_file_paths)
        """
        if not self.vault_path.exists():
            return "Obsidian Vault directory does not exist.", []

        # Handle both pre-extracted keyword lists and raw string queries
        if isinstance(query, list):
            keywords = [k.lower().strip() for k in query if k and k.strip()]
        elif isinstance(query, str):
            keywords = [k.lower().strip() for k in query.split() if len(k.strip()) > 2]
        else:
            return "Invalid search query type provided.", []

        if not keywords:
            return "Empty or invalid search query provided.", []

        scored_matches = []  # List of tuples: (score, match_block, rel_path_str)

        for md_file in self.vault_path.rglob("*.md"):
            # Ignore configuration, git, archive, or log directories
            if any(part in {".obsidian", ".git", "40_Archives", "50_Logs"} for part in md_file.parts):
                continue

            try:
                content = md_file.read_text(encoding="utf-8")
                rel_path = md_file.relative_to(self.vault_path)
                rel_path_str = str(rel_path).replace("\\", "/")
                filename_lower = md_file.stem.lower()

                score = 0

                # 1. Title match scoring (Highest priority)
                for kw in keywords:
                    if kw in filename_lower:
                        score += 10

                # 2. YAML frontmatter / Tag match scoring (only if file begins with ---)
                lines = content.splitlines()
                if lines and lines[0].strip() == "---":
                    for line in lines[1:20]:
                        line_lower = line.lower()
                        if line_lower.startswith("---"):
                            break
                        if any(kw in line_lower for kw in keywords):
                            score += 5

                # 3. Body content match scoring
                matching_indices = [
                    idx for idx, line in enumerate(lines)
                    if any(kw in line.lower() for kw in keywords)
                ]
                score += len(matching_indices)

                # Collect contextual snippet if file has any match relevance
                if score > 0:
                    if len(lines) <= 60 and len(content) <= 2500:
                        snippet = content.strip()
                    elif matching_indices:
                        # Include surrounding context windows (e.g., code blocks) around top matches
                        included_indices: set[int] = set()
                        for match_idx in matching_indices[:6]:
                            start = max(0, match_idx - 2)
                            end = min(len(lines), match_idx + 8)
                            included_indices.update(range(start, end))
                        ordered = sorted(included_indices)[:60]
                        snippet = "\n".join(lines[i] for i in ordered)
                    else:
                        snippet = "\n".join(lines[:30])

                    match_block = f"--- Note: [[{rel_path_str}]] (Relevance Score: {score}) ---\n{snippet}"
                    scored_matches.append((score, match_block, rel_path_str))

            except Exception:
                continue

        if not scored_matches:
            return (
                f"No relevant notes found in Obsidian Vault for query: '{query}'.",
                [],
            )

        # Sort matches by relevance score in descending order
        scored_matches.sort(key=lambda x: x[0], reverse=True)

        top_matches = scored_matches[:max_results]
        formatted_text = "\n\n".join([m[1] for m in top_matches])
        matched_sources = [f"{m[2]} (Score: {m[0]})" for m in top_matches]
        
        return formatted_text, matched_sources

    def search(self, query: str, max_results: int = 5) -> str:
        """Search across all Markdown notes in the Obsidian Vault for matching keywords.
        Backward-compatible wrapper returning only the formatted text string.
        """
        result_text, _ = self.search_with_sources(query, max_results=max_results)
        return result_text

    def read_rules(
        self,
        task_instruction: str = "",
        relative_rules_folder: str = "30_Resources/AgentRules",
    ) -> str:
        """Read core engineering rules unconditionally, and append domain-specific rules
        only if matching keywords are found in the task instruction.
        """
        if not self.vault_path.exists():
            return "No specific rules found."

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

        # Sanitize title for file system compatibility
        clean_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")
        clean_title = re.sub(r"_+", "_", clean_title)
        filename = f"{date_prefix}_{clean_title}.md"

        # Strip redundant outer markdown code block wrappers
        cleaned_content = content.strip()
        match = re.search(
            r"^```(?:markdown)?\s*([\s\S]*?)\s*```$", cleaned_content, re.IGNORECASE
        )
        if match:
            cleaned_content = match.group(1).strip()

        # Normalize YAML frontmatter indentation to prevent Obsidian parsing errors
        if cleaned_content.startswith("---"):
            parts = cleaned_content.split("---", 2)
            if len(parts) >= 3:
                yaml_part = "\n".join(
                    line.rstrip() for line in parts[1].strip().splitlines()
                )
                body_part = parts[2].lstrip()
                cleaned_content = f"---\n{yaml_part}\n---\n\n{body_part}"

        file_path = target_dir / filename
        file_path.write_text(cleaned_content, encoding="utf-8")
        return file_path