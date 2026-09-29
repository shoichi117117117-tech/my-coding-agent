"""Tester Agent module responsible for automated test generation, RCA diagnosis, and Obsidian knowledge archiving."""

import json
import re
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.base import BaseAgent, extract_response_content
from tools.code_editor import CodeEditor
from tools.obsidian import ObsidianAdapter
from utils.ui import AgentLogger

TESTER_SYSTEM_PROMPT = """You are the Lead QA & Knowledge Engineer (Tester Agent) responsible for inspecting code, generating test suites, analyzing execution logic, and providing diagnostic RCA feedback for the Planner Agent.

Current Workspace Root: {workspace_root}
Obsidian Vault Path: {obsidian_vault_path}

Strict Operational Rules:
1. Dynamic Knowledge Base Consultation:
   - BEFORE writing complex test suites or diagnosing module resolution failures, YOU MUST use the `search_obsidian_vault` tool if you need to check framework-specific testing rules (e.g. Pytest options, Jest setup) or historical solution notes.

2. Code Quality & Test Verification:
   - Carefully review code generated or modified by the Coder Agent.
   - For Python projects, test files MUST ALWAYS be saved inside the `tests/` directory (e.g., `tests/test_main.py`).
   - Generate runnable unit tests or verification scripts matching the workspace tech stack.

3. Diagnostic Feedback for Planner Agent (Root Cause Analysis):
   - When execution failures occur, inspect code and configuration files (`pyproject.toml`, `package.json`, etc.) to conduct Root Cause Analysis (RCA).
   - Provide structured, actionable RCA diagnostic reports so the Planner can formulate concrete correction plans.

4. Obsidian Knowledge Archival:
   - When formatting resolved errors into Obsidian notes, strictly use YAML Frontmatter (`type: troubleshooting`, `date`, `tags`, `aliases`).
   - Use Obsidian WikiLinks (e.g., `[[Python]]` or `[[Testing-Guide]]`) to connect knowledge nodes.

5. Language & Documentation Standard:
   - ALL inline comments, docstrings, and technical reports MUST be written in ENGLISH.

6. Tools Usage:
   - Use `search_obsidian_vault` to query internal documentation and testing guidelines.
   - Use workspace manipulation tools (`read_file`, `search_files`, `write_file`, `delete_file`) to inspect workspace files and write test scripts safely.
"""

TESTER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_obsidian_vault",
            "description": "Search the local Obsidian Knowledge Vault for testing guidelines, manifest schemas, or past troubleshooting notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keyword query to search in Obsidian Vault (e.g., 'pytest options', 'troubleshooting fastapi')",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the content of a target file to analyze, verify, and test.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Relative path to the file from workspace root"}
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for specific test files or code implementations within the workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "Keyword or file pattern to search (e.g., '.py' or 'test_')"}
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or overwrite content to a test file or documentation note safely.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path to the file/note to create or overwrite",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full Markdown or code content to write",
                    },
                },
                "required": ["file_path"],
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


class TesterAgent(BaseAgent):
    """Tester Agent providing unit test generation, RCA reports, and troubleshooting note creation."""

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        obsidian_vault_path: Optional[str] = None,
        model_fallback_list: Optional[List[str]] = None,
    ) -> None:
        """Initialize TesterAgent with workspace tools and obsidian integration."""
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else None
        self.obsidian_vault_path = Path(obsidian_vault_path).resolve() if obsidian_vault_path else None

        self.code_editor = CodeEditor(workspace_root=str(self.workspace_root)) if self.workspace_root else None
        self.obsidian_adapter = ObsidianAdapter(str(self.obsidian_vault_path)) if self.obsidian_vault_path else None
        self.ui_logger = AgentLogger("Tester")

        formatted_system_prompt = TESTER_SYSTEM_PROMPT.format(
            workspace_root=str(self.workspace_root) if self.workspace_root else "Not specified",
            obsidian_vault_path=str(self.obsidian_vault_path) if self.obsidian_vault_path else "Not specified",
        )

        super().__init__(
            name="Tester",
            system_prompt=formatted_system_prompt,
            role="tester",
            model_fallback_list=model_fallback_list,
        )

    def extract_pure_code(self, raw_text: str) -> str:
        """Extract pure code string by removing markdown backtick code blocks."""
        text = raw_text.strip()
        match = re.search(r"```(?:\w+)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text

    def generate_test_code(
        self,
        target_file: str,
        code_content: str,
        test_requirements: str = "",
        auto_fetch_vault_rules: bool = True,
    ) -> str:
        """Generate automated unit test code targeting specified codebase files."""
        self.ui_logger.start_action(f"Generating test suite for target file: '{target_file}'...")
        vault_rules_context = ""
        if auto_fetch_vault_rules and self.obsidian_adapter:
            vault_rules_context = self.obsidian_adapter.read_rules(task_instruction=test_requirements)

        user_prompt = (
            f"Generate comprehensive unit tests for the following source file.\n\n"
            f"Target File: {target_file}\n"
            f"Code Content:\n```\n{code_content}\n```\n\n"
            f"Requirements: {test_requirements if test_requirements else 'Cover normal cases, edge cases, and error handling.'}\n"
            f"Obsidian Rules/Guidelines:\n{vault_rules_context if vault_rules_context else 'None'}\n\n"
            f"Rule: Return ONLY executable test code in a standard markdown code block."
        )

        messages = [{"role": "user", "content": user_prompt}]
        response = self.call_llm(messages=messages)
        raw_content = extract_response_content(response)
        test_code = self.extract_pure_code(raw_content)

        self.ui_logger.stop_action(f"Unit test code generated successfully for '{target_file}'.")
        return test_code

    def _execute_tool_call(self, tool_call: Any) -> str:
        """Execute dispatched tool actions for file manipulation and vault retrieval."""
        func_name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except Exception as e:
            return f"Error parsing arguments: {e}"

        target_summary = args.get("file_path") or args.get("query") or ""
        self.ui_logger.tool_call(func_name, target_summary)

        if func_name == "search_obsidian_vault":
            if not self.obsidian_adapter:
                return "Error: Obsidian Vault adapter is not configured."
            return self.obsidian_adapter.search(query=args.get("query", ""))

        if not self.code_editor:
            return "Error: CodeEditor is not initialized."

        tool_method = getattr(self.code_editor, func_name, None)
        if not tool_method:
            return f"Error: Tool '{func_name}' is not recognized or supported."

        try:
            if func_name == "write_file" and "content" in args:
                args["content"] = self.extract_pure_code(args["content"])
            result = tool_method(**args)
            return str(result)
        except Exception as e:
            self.ui_logger.error(f"Tool execution failed [{func_name}]: {e}")
            return f"Error executing {func_name}: {e}"

    def execute_task(
        self,
        task_description: str,
        target_file: str = "",
        execution_log: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        max_tool_iterations: int = 5,
        auto_fetch_vault_rules: bool = True,
    ) -> Any:
        """Perform diagnostic RCA or verification task using tool calls."""
        messages = list(history) if history else []

        vault_rules_context = ""
        if auto_fetch_vault_rules and self.obsidian_adapter:
            vault_rules_context = self.obsidian_adapter.read_rules(task_instruction=task_description)

        user_message_parts = [
            f"【Target File for Testing/Verification】\n{target_file if target_file else 'Not specified'}",
            f"【Test Task Requirements】\n{task_description}",
        ]

        if vault_rules_context:
            user_message_parts.append(
                f"【Obsidian Vault Rules & Guidelines】\n{vault_rules_context}"
            )

        if execution_log:
            user_message_parts.append(
                f"【Terminal / Test Execution Log】\n```text\n{execution_log.strip()}\n```"
            )

        user_message_parts.append(
            "Inspect target workspace files using tools if needed. "
            "Use `search_obsidian_vault` if you need testing conventions or solution rules. "
            "If an error or failure occurred, perform Root Cause Analysis (RCA) "
            "and output a structured diagnostic report for the Planner Agent."
        )

        user_message = "\n\n".join(user_message_parts)
        messages.append({"role": "user", "content": user_message})

        self.ui_logger.start_action(f"Running verification & diagnosis: {task_description[:50]}...")

        for iteration in range(max_tool_iterations):
            response = self.call_llm(messages=messages, tools=TESTER_TOOLS)
            message = response.choices[0].message if hasattr(response, "choices") else response

            raw_content = extract_response_content(message)
            msg_dict = {"role": "assistant", "content": raw_content}
            if hasattr(message, "tool_calls") and message.tool_calls:
                msg_dict["tool_calls"] = message.tool_calls
            messages.append(msg_dict)

            if not hasattr(message, "tool_calls") or not message.tool_calls:
                self.ui_logger.stop_action("Verification task completed.")
                return raw_content

            for tool_call in message.tool_calls:
                tool_result = self._execute_tool_call(tool_call)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })

        self.ui_logger.stop_action("Verification task completed after tool iterations.")
        return response

    def generate_obsidian_troubleshooting_note(
        self,
        error_title: str,
        error_log: str,
        solution_summary: str,
        save_to_vault: bool = True,
        relative_folder: str = "30_Resources/Troubleshooting",
    ) -> str:
        """Generate structured Markdown troubleshooting note and save to Obsidian Vault."""
        self.ui_logger.start_action(f"Archiving troubleshooting knowledge note: '{error_title}'...")
        today_str = datetime.now().strftime("%Y-%m-%d")

        user_prompt = (
            f"Analyze the following resolved issue and provide detailed content for a troubleshooting note:\n\n"
            f"Title: {error_title}\n"
            f"Error Output:\n```text\n{error_log}\n```\n"
            f"Solution Applied:\n{solution_summary}\n\n"
            f"Instructions:\n"
            f"1. Explain the Root Cause Analysis (RCA) in detail under a '## 2. Root Cause Analysis' section.\n"
            f"2. Summarize the Solution & Prevention Rules clearly under a '## 3. Solution & Prevention Rules' section.\n"
            f"3. Use Obsidian WikiLinks (e.g., [[Python]], [[Pytest]], [[FastAPI]]) where appropriate.\n"
            f"4. Do NOT include Frontmatter (---) or the main title (# Title). Start directly with the '## 2. Root Cause Analysis' header."
        )

        messages = [{"role": "user", "content": user_prompt}]
        response = self.call_llm(messages=messages)
        body_content = extract_response_content(response).strip()

        match = re.search(r"^```(?:markdown)?\s*([\s\S]*?)\s*```$", body_content, re.IGNORECASE)
        if match:
            body_content = match.group(1).strip()

        markdown_content = textwrap.dedent(f"""\
            ---
            type: troubleshooting
            date: '{today_str}'
            tags:
              - error-solution
              - troubleshooting
            ---

            # [Issue] {error_title}

            ## 1. Symptom & Error Log
            ```text
            {error_log.strip()}
            ```

            {body_content}
        """)

        if save_to_vault and self.obsidian_adapter:
            try:
                saved_path = self.obsidian_adapter.save_troubleshooting_note(
                    title=error_title,
                    content=markdown_content,
                    relative_folder=relative_folder,
                )
                self.ui_logger.stop_action(f"Note successfully archived to Vault at: '{saved_path}'")
            except Exception as e:
                self.ui_logger.error(f"Failed to save note to Obsidian Vault: {e}")

        return markdown_content