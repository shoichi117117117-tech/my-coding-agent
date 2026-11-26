"""Coder Agent module responsible for writing source code and managing project manifests across stacks."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from my_coding_agent.agents.base import BaseAgent, extract_response_content
from my_coding_agent.tools.code_editor import CodeEditor
from my_coding_agent.tools.obsidian import ObsidianAdapter
from my_coding_agent.utils.ui import AgentLogger

CODER_SYSTEM_PROMPT = """You are the Coder Agent responsible for implementing, refactoring, and maintaining production-grade code and project configuration files across workspace files.

Current Workspace Root: {workspace_root}
Obsidian Vault Root: {obsidian_vault_path}

Strict Operational Rules & Knowledge Retrieval Strategy:
1. MANDATORY Knowledge Base Consultation (Overcoming Pre-trained Memory Overconfidence):
   - BEFORE creating or updating any project configuration manifests (`pyproject.toml`, `package.json`, `Cargo.toml`, `pom.xml`, etc.), writing framework-specific entry points (FastAPI, React), or configuring test options (Pytest, Jest):
     YOU MUST NOT rely solely on your pre-trained internal memory.
   - YOU MUST ALWAYS call the `search_obsidian_vault` tool FIRST using relevant keywords (e.g., 'pyproject.toml', 'pytest config', 'package.json schema') to retrieve ground-truth project guidelines.

2. Agentic Execution Workflow (Plan-Before-Act Protocol):
   - STEP 1 (THOUGHT & KNOWLEDGE RETRIEVAL): State a brief 1-line reasoning regarding whether project guidelines or schema rules are needed. If the task involves setup or manifest files, execute `search_obsidian_vault` FIRST before taking modifying actions.
   - STEP 2 (IMPLEMENTATION): Review retrieved guidelines from tool outputs and write clean code/configs adhering strictly to project standard schemas.

3. Manifest Generation & Schema Compliance:
   - Always create or update necessary project configuration manifests strictly aligned with the retrieved target language schema and Planner's instructions.
   - For Python projects using `uv`, strictly adhere to standard PEP 621 (`[project]`) and PEP 735 (`[dependency-groups]`) specifications. NEVER generate vendor-specific configurations like `[tool.poetry]` unless explicitly commanded in the Vault guideline.
   - **CRITICAL TOML & HATCHLING RULES**:
     - NEVER write `[project.dependencies]` as a separate table header, and NEVER nest `dependency-groups` inside `[project]`.
     - If `[project]` includes `readme = "README.md"`, you MUST ALWAYS create `README.md` via `write_file` in the same turn; otherwise `hatchling` fails with `OSError: Readme file does not exist: README.md`.
     - For flat script/module layouts (e.g., `main.py` or `slugify.py` in the workspace root without a package subfolder), OMIT `[build-system]` and set `[tool.uv]\npackage = false` so `uv sync` does not fail trying to build an editable wheel.
     - Only use `[build-system]` with `hatchling.build` when creating an actual Python package directory (e.g., `<pkg_name>/__init__.py` or `src/<pkg_name>/__init__.py` with matching `[tool.hatch.build.targets.wheel] packages = [...]`), AND configure `[tool.pytest.ini_options]` (`pythonpath`, `testpaths`) to match test imports.
     - Dependencies MUST always be written as an array inside `[project]`:
       ```toml
       [project]
       dependencies = [
           "pytest>=8.0.0",
       ]
       ```

4. Pre-Flight Validation & Tool-Error Self-Correction:
   - If any tool (such as `write_file`) returns a "PRE-FLIGHT SYNTAX ERROR", "PRE-FLIGHT VALIDATION FAILED", or "⚠️ WORKSPACE WARNINGS FOR `uv sync`" response:
     DO NOT proceed to ending your turn.
   - Read the exact error or warning details returned by the tool and resolve all missing files (e.g., `README.md`, package directory) or configuration mismatches immediately before finishing.

5. Directory Rules & Verification Readiness:
   - For Python projects, unit test files MUST ALWAYS be written inside a `tests/` directory (e.g., `tests/test_main.py`).
   - Write clean, production-grade, fully executable code without placeholder stubs.
   - For standalone scripts or API services, ensure entry points work non-interactively.

6. Language & Documentation Standard:
   - ALL inline code comments, docstrings, and technical descriptions MUST be written in ENGLISH.

7. Tool Usage:
   - Use `search_obsidian_vault` to query internal documentation and rules.
   - Use workspace manipulation tools (`search_files`, `read_file`, `write_file`, `make_directory`, `delete_file`) to inspect workspace files and apply modifications accurately.
"""

CODER_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_obsidian_vault",
            "description": "Search the local Obsidian Knowledge Vault for project guidelines, manifest schemas (e.g., 'pyproject.toml', 'pytest', 'package.json'), coding conventions, or past troubleshooting notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keyword query to search in Obsidian Vault (e.g., 'pyproject.toml format', 'pytest rules', 'fastapi structure')",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files matching a keyword or pattern within the workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Keyword or file pattern to search (e.g., '.py', 'package.json', or 'src')",
                    }
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the text content of a target file safely.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path to the target file from the workspace root",
                    }
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or overwrite content to a target file safely. Automatically executes pre-flight syntax validation for Python, TOML, and JSON files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path to the file to create or overwrite",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full file content to write",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_directory",
            "description": "Create target directory hierarchy safely within workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {
                        "type": "string",
                        "description": "Relative path of directory to create",
                    }
                },
                "required": ["dir_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Delete an obsolete or unnecessary file safely within the workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path to the file to delete",
                    }
                },
                "required": ["file_path"],
            },
        },
    },
]


class CoderAgent(BaseAgent):
    """Coder Agent responsible for writing source code and project configuration manifests."""

    def __init__(
        self,
        workspace_root: Optional[Union[str, Path]] = None,
        obsidian_vault_path: Optional[Union[str, Path]] = None,
        model_fallback_list: Optional[List[str]] = None,
    ) -> None:
        """Initialize CoderAgent with file editor and obsidian vault tools."""
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else Path.cwd()
        self.obsidian_vault_path = Path(obsidian_vault_path).resolve() if obsidian_vault_path else None

        self.code_editor = CodeEditor(workspace_root=str(self.workspace_root))
        self.obsidian_adapter = ObsidianAdapter(str(self.obsidian_vault_path)) if self.obsidian_vault_path else None
        self.ui_logger = AgentLogger("Coder")

        formatted_system_prompt = CODER_SYSTEM_PROMPT.format(
            workspace_root=str(self.workspace_root),
            obsidian_vault_path=str(self.obsidian_vault_path) if self.obsidian_vault_path else "Not specified",
        )

        super().__init__(
            name="Coder",
            system_prompt=formatted_system_prompt,
            role="coder",
            model_fallback_list=model_fallback_list,
        )

    def extract_pure_code(self, raw_text: str) -> str:
        """Extract clean code string, stripping an outer markdown code block wrapper if present."""
        text = raw_text.strip()
        match = re.match(r"^```(?:\w+)?\s*\n([\s\S]*?)\n?```$", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text

    def _execute_tool_call(self, tool_call: Any) -> str:
        """Dispatch tool calls to Obsidian adapter or CodeEditor instance."""
        func_name = tool_call.function.name
        raw_args = tool_call.function.arguments
        try:
            if isinstance(raw_args, dict):
                args = raw_args
            elif not raw_args:
                args = {}
            else:
                args = json.loads(raw_args)
        except Exception as e:
            return f"Error parsing arguments for tool '{func_name}': {e}"

        target_summary = str(args.get("file_path") or args.get("dir_path") or args.get("query") or args.get("keyword") or "")
        self.ui_logger.tool_call(func_name, target_summary)

        if func_name == "search_obsidian_vault":
            if not self.obsidian_adapter:
                return "Error: Obsidian Vault adapter is not configured."

            raw_query = args.get("query", "")
            keywords = self._extract_search_keywords(raw_query)
            content, read_sources = self.obsidian_adapter.search_with_sources(query=keywords)

            if read_sources:
                self.ui_logger.tool_result(read_sources)
            return content

        tool_method = getattr(self.code_editor, func_name, None)
        if not tool_method:
            return f"Error: Tool '{func_name}' is not recognized or supported."

        try:
            if func_name == "write_file" and "content" in args:
                file_path = str(args.get("file_path", "")).lower()
                if not file_path.endswith(".md"):
                    args["content"] = self.extract_pure_code(args["content"])
            result = tool_method(**args)
            return str(result)
        except Exception as e:
            self.ui_logger.error(f"Tool execution failed [{func_name}]: {e}")
            return f"Error executing {func_name}: {e}"

    def execute_task(
        self,
        task_description: str,
        rules: str = "",
        error_context: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        max_tool_iterations: int = 5,
        auto_fetch_vault_rules: bool = True,
        iteration_current: int = 1,
        iteration_total: int = 1,
    ) -> Any:
        """Execute implementation tasks utilizing interactive tool call loops."""
        messages: List[Dict[str, Any]] = list(history) if history else []

        if auto_fetch_vault_rules and self.obsidian_adapter:
            vault_rules = self.obsidian_adapter.read_rules(task_instruction=task_description[:100])
            active_rules = f"{rules}\n\n{vault_rules}".strip()
        else:
            active_rules = rules.strip() if rules else "No specific rules provided."

        user_message_parts = [
            f"【Active Development & Obsidian Knowledge Rules】\n{active_rules}",
            f"【Target Sub-Task】\n{task_description}",
        ]

        if error_context:
            user_message_parts.append(
                f"【Previous Execution Error / Failure Context】\n```text\n{error_context.strip()}\n```"
            )

        user_message_parts.append(
            "CRITICAL DIRECTIVE: If your task involves setting up configuration files (e.g., pyproject.toml, package.json) "
            "or framework structures, execute `search_obsidian_vault` tool FIRST to fetch project rules. "
            "Do NOT rely on assumptions."
        )

        user_message = "\n\n".join(user_message_parts)
        messages.append({"role": "user", "content": user_message})

        is_config_task = any(
            kw in task_description.lower() for kw in ["pyproject", "toml", "package.json", "config", "manifest", "setup"]
        )

        self.ui_logger.info("Implementing code modifications...")

        with self.ui_logger.status("Implementing code modifications..."):
            for iteration in range(max_tool_iterations):
                if iteration == 0 and is_config_task and self.obsidian_adapter:
                    iteration_messages = messages + [{
                        "role": "user",
                        "content": (
                            "SYSTEM MANDATE: Before writing any file, call `search_obsidian_vault` FIRST "
                            "to inspect project schema guidelines and rules."
                        )
                    }]
                else:
                    iteration_messages = messages

                response = self.call_llm(messages=iteration_messages, tools=CODER_TOOLS)
                message = response.choices[0].message if hasattr(response, "choices") else response

                raw_content = extract_response_content(message)

                msg_dict: Dict[str, Any] = {"role": "assistant", "content": raw_content}
                if hasattr(message, "tool_calls") and message.tool_calls:
                    msg_dict["tool_calls"] = message.tool_calls
                messages.append(msg_dict)

                if not hasattr(message, "tool_calls") or not message.tool_calls:
                    self.ui_logger.stop_action("Code implementation completed.")
                    return response

                for tool_call in message.tool_calls:
                    tool_result = self._execute_tool_call(tool_call)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    })

        self.ui_logger.stop_action("Code implementation completed.")
        return response