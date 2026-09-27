import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from agents.base import BaseAgent, extract_response_content
from tools.code_editor import CodeEditor
from tools.obsidian import ObsidianAdapter

logger = logging.getLogger(__name__)

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
   - For Python projects using `uv`, strictly adhere to standard PEP 621 (`[project]`) specifications. NEVER generate vendor-specific configurations like `[tool.poetry]` unless explicitly commanded in the Vault guideline.

4. Directory Rules & Verification Readiness:
   - For Python projects, unit test files MUST ALWAYS be written inside a `tests/` directory (e.g., `tests/test_main.py`).
   - Write clean, production-grade, fully executable code without placeholder stubs.
   - For standalone scripts or API services, ensure entry points work non-interactively.

5. Language & Documentation Standard:
   - ALL inline code comments, docstrings, and technical descriptions MUST be written in ENGLISH.

6. Tool Usage:
   - Use `search_obsidian_vault` to query internal documentation and rules.
   - Use workspace manipulation tools (`search_files`, `read_file`, `write_file`, `make_directory`, `delete_file`) to inspect workspace files and apply modifications accurately.
"""

CODER_TOOLS = [
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
            "description": "Write or overwrite content to a target file safely.",
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
    """Generic Coder Agent responsible for writing code and project configs across language stacks."""

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        obsidian_vault_path: Optional[str] = None,
        model_fallback_list: Optional[List[str]] = None,
    ):
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else Path.cwd()
        self.obsidian_vault_path = Path(obsidian_vault_path).resolve() if obsidian_vault_path else None

        self.code_editor = CodeEditor(workspace_root=str(self.workspace_root))
        self.obsidian_adapter = ObsidianAdapter(str(self.obsidian_vault_path)) if self.obsidian_vault_path else None

        formatted_system_prompt = CODER_SYSTEM_PROMPT.format(
            workspace_root=str(self.workspace_root),
            obsidian_vault_path=str(self.obsidian_vault_path) if self.obsidian_vault_path else "Not specified",
        )

        super().__init__(
            name="Coder",
            system_prompt=formatted_system_prompt,
            model_fallback_list=model_fallback_list,
        )

    def extract_pure_code(self, raw_text: str) -> str:
        """Extract pure code content stripping markdown triple backticks if present."""
        text = raw_text.strip()
        match = re.search(r"```(?:\w+)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text

    def _execute_tool_call(self, tool_call: Any) -> str:
        """Helper method to dispatch tool execution directly to instance methods."""
        func_name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except Exception as e:
            return f"Error parsing arguments: {e}"

        # Dispatch Obsidian Vault Search Tool
        if func_name == "search_obsidian_vault":
            if not self.obsidian_adapter:
                return "Error: Obsidian Vault adapter is not configured."
            query = args.get("query", "")
            return self.obsidian_adapter.search(query=query)

        # Dispatch Workspace CodeEditor Tools
        tool_method = getattr(self.code_editor, func_name, None)
        if not tool_method:
            return f"Error: Tool '{func_name}' is not recognized or supported."

        try:
            if func_name == "write_file" and "content" in args:
                args["content"] = self.extract_pure_code(args["content"])
            result = tool_method(**args)
            return str(result)
        except Exception as e:
            logger.error(f"[Coder Tool Execution Error] {func_name}: {e}")
            return f"Error executing {func_name}: {e}"

    def execute_task(
        self,
        task_description: str,
        rules: str = "",
        error_context: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        max_tool_iterations: int = 5,
        auto_fetch_vault_rules: bool = True,
    ) -> Any:
        """Executes coding or configuration sub-tasks using dynamic tool calls and Obsidian knowledge retrieval."""
        messages = list(history) if history else []

        # Fetch initial core Obsidian rules dynamically if enabled
        if auto_fetch_vault_rules and self.obsidian_adapter:
            vault_rules = self.obsidian_adapter.read_rules(task_instruction=task_description)
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
            "CRITICAL DIRECTIVE: If your task involves setting up configuration files (e.g. pyproject.toml, package.json) "
            "or framework structures, execute `search_obsidian_vault` tool FIRST to fetch project rules. "
            "Do NOT rely on assumptions."
        )

        user_message = "\n\n".join(user_message_parts)
        messages.append({"role": "user", "content": user_message})

        logger.info(f"[Coder] Processing task in workspace '{self.workspace_root}': {task_description[:60]}...")

        # Detect configuration/manifest setup intent to enable dynamic priming
        is_config_task = any(
            kw in task_description.lower() for kw in ["pyproject", "toml", "package.json", "config", "manifest", "setup"]
        )

        for iteration in range(max_tool_iterations):
            # Dynamic Context Injection: Prompt priming on the first iteration for config tasks
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
            msg_dict = {"role": "assistant", "content": raw_content}
            if hasattr(message, "tool_calls") and message.tool_calls:
                msg_dict["tool_calls"] = message.tool_calls
            messages.append(msg_dict)

            if not hasattr(message, "tool_calls") or not message.tool_calls:
                return response

            for tool_call in message.tool_calls:
                logger.info(f"[Coder] Tool Call requested: {tool_call.function.name}")
                tool_result = self._execute_tool_call(tool_call)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })

        return response