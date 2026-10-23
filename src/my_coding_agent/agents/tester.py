"""Tester Agent module responsible for automated test generation, RCA diagnosis, and Obsidian knowledge archiving."""

import json
import re
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from my_coding_agent.agents.base import BaseAgent, extract_response_content
from my_coding_agent.tools.code_editor import CodeEditor
from my_coding_agent.tools.obsidian import ObsidianAdapter
from my_coding_agent.utils.ui import AgentLogger

TESTER_SYSTEM_PROMPT = """You are the Lead QA & Knowledge Engineer (Tester Agent) responsible for inspecting code, generating test suites, analyzing execution logic, and providing diagnostic RCA feedback for the Planner Agent.

Current Workspace Root: {workspace_root}
Obsidian Vault Path: {obsidian_vault_path}

Strict Operational Rules:
1. Dynamic Knowledge Base Consultation:
   - BEFORE writing complex test suites or diagnosing module resolution failures, YOU MUST use the `search_obsidian_vault` tool if you need to check framework-specific testing rules (e.g., Pytest options, Jest setup) or historical solution notes.

2. Code Quality & Test Verification:
   - Carefully review code generated or modified by the Coder Agent.
   - For Python projects, test files MUST ALWAYS be saved inside the `tests/` directory (e.g., `tests/test_main.py`).
   - Generate runnable unit tests or verification scripts matching the workspace tech stack.

3. Diagnostic Feedback for Planner Agent (Root Cause Analysis):
   - When execution failures occur, inspect code and configuration files (`pyproject.toml`, `package.json`, etc.) to conduct Root Cause Analysis (RCA).
   - Provide structured, actionable RCA diagnostic reports so the Planner can formulate concrete correction plans.

4. Pre-Flight Tool Error Awareness:
   - If a tool returns a syntax validation failure message (e.g., "PRE-FLIGHT SYNTAX ERROR"), inspect the file content and provide clear, corrective instructions in your analysis regarding syntax compliance.

5. Obsidian Knowledge Archival:
   - When formatting resolved errors into Obsidian notes, strictly use YAML Frontmatter (`type: troubleshooting`, `date`, `tags`, `aliases`).
   - Use Obsidian WikiLinks (e.g., `[[Python]]` or `[[Testing-Guide]]`) to connect knowledge nodes.

6. Language & Documentation Standard:
   - ALL inline comments, docstrings, and technical reports MUST be written in ENGLISH.

7. Tools Usage:
   - Use `search_obsidian_vault` to query internal documentation and testing guidelines.
   - Use workspace manipulation tools (`read_file`, `search_files`, `write_file`, `delete_file`) to inspect workspace files and write test scripts safely.
"""

TESTER_TOOLS: List[Dict[str, Any]] = [
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
            "description": "Write or overwrite content to a test file or documentation note safely. Executes pre-flight syntax checks on supported file formats.",
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
                "required": ["file_path", "content"],
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


class TroubleshootingAnalysis(BaseModel):
    """Pydantic schema enforcing structured JSON output from the LLM."""

    tags: List[str] = Field(
        description="Extract 2-5 relevant technology or domain tags (e.g., ['python', 'venv', 'pytest'])."
    )
    related_topics: List[str] = Field(
        description="Extract 1-3 core topic keywords for Obsidian WikiLinks without brackets (e.g., ['Python', 'Virtual Environment'])."
    )
    root_cause_analysis: str = Field(
        description="Detailed explanation for the '## 2. Root Cause Analysis' section."
    )
    solution_and_prevention: str = Field(
        description="Clear actionable steps for the '## 3. Solution & Prevention Rules' section."
    )


class TesterAgent(BaseAgent):
    """Tester Agent providing unit test generation, RCA reports, and troubleshooting note creation."""

    __test__ = False

    def __init__(
        self,
        workspace_root: Optional[Union[str, Path]] = None,
        obsidian_vault_path: Optional[Union[str, Path]] = None,
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
        self.ui_logger.info(f"Generating unit test suite for {target_file}...")
        
        with self.ui_logger.status("Drafting unit tests via LLM..."):
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

            return test_code

    def _execute_tool_call(self, tool_call: Any) -> str:
        """Execute dispatched tool actions for file manipulation and vault retrieval."""
        func_name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except Exception as e:
            return f"Error parsing arguments for tool '{func_name}': {e}"

        target_summary = str(args.get("file_path") or args.get("query") or "")
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
        """Perform diagnostic RCA or verification task using tool call iterations."""
        messages: List[Dict[str, Any]] = list(history) if history else []

        with self.ui_logger.status(f"Running verification & diagnosis: {task_description[:50]}..."):
            vault_rules_context = ""
            if auto_fetch_vault_rules and self.obsidian_adapter:
                vault_rules_context = self.obsidian_adapter.read_rules(task_instruction=task_description)

            user_message_parts = [
                f"【Target File for Testing/Verification】\n{target_file if target_file else 'Not specified'}",
                f"【Test Task Requirements】\n{task_description}",
            ]

            if vault_rules_context:
                user_message_parts.append(f"【Obsidian Vault Rules & Guidelines】\n{vault_rules_context}")

            if execution_log:
                user_message_parts.append(f"【Terminal / Test Execution Log】\n```text\n{execution_log.strip()}\n```")

            user_message_parts.append(
                "Inspect target workspace files using tools if needed. "
                "Use `search_obsidian_vault` if you need testing conventions or solution rules. "
                "If an error or failure occurred, perform Root Cause Analysis (RCA) "
                "and output a structured diagnostic report for the Planner Agent."
            )

            user_message = "\n\n".join(user_message_parts)
            messages.append({"role": "user", "content": user_message})

            for _ in range(max_tool_iterations):
                response = self.call_llm(messages=messages, tools=TESTER_TOOLS)
                message = response.choices[0].message if hasattr(response, "choices") else response

                raw_content = extract_response_content(message)

                msg_dict: Dict[str, Any] = {"role": "assistant", "content": raw_content}
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
        """Generate structured Markdown troubleshooting note using Pydantic structured output, bypassing regex parsing."""
        with self.ui_logger.status(
            f"Archiving troubleshooting knowledge note: '{error_title}'..."
        ):
            today_str = datetime.now().strftime("%Y-%m-%d")

            user_prompt = (
                f"Analyze the following resolved issue and extract structured knowledge for an Obsidian note.\n\n"
                f"Title: {error_title}\n"
                f"Error Output:\n```text\n{error_log}\n```\n"
                f"Solution Applied:\n{solution_summary}"
            )

            messages = [{"role": "user", "content": user_prompt}]

            structured_data: TroubleshootingAnalysis = self.call_llm_structured(
                messages=messages, response_format=TroubleshootingAnalysis
            )

            tags = ["troubleshooting", "error-solution"]
            sanitized_extracted_tags = [
                t.strip().lower().replace(" ", "-")
                for t in structured_data.tags
                if t.strip()
            ]
            tags.extend(sanitized_extracted_tags)
            unique_tags = list(dict.fromkeys(tags))

            related_topics = [
                f'"[[{topic.strip()}]]"'
                for topic in structured_data.related_topics
                if topic.strip()
            ] or ['"[[Troubleshooting]]"']

            tags_yaml = "\n".join([f"  - {tag}" for tag in unique_tags])
            topics_yaml = "\n".join([f"  - {topic}" for topic in related_topics])

            markdown_content = (
                f"---\n"
                f"type: troubleshooting\n"
                f"date: '{today_str}'\n"
                f"tags:\n"
                f"{tags_yaml}\n"
                f"related_topics:\n"
                f"{topics_yaml}\n"
                f"---\n\n"
                f"# [Issue] {error_title}\n\n"
                f"## 1. Symptom & Error Log\n"
                f"```text\n"
                f"{error_log.strip()}\n"
                f"```\n\n"
                f"## 2. Root Cause Analysis\n"
                f"{structured_data.root_cause_analysis.strip()}\n\n"
                f"## 3. Solution & Prevention Rules\n"
                f"{structured_data.solution_and_prevention.strip()}\n"
            )

            if save_to_vault and self.obsidian_adapter:
                try:
                    saved_path = self.obsidian_adapter.save_troubleshooting_note(
                        title=error_title,
                        content=markdown_content,
                        relative_folder=relative_folder,
                    )
                    self.ui_logger.stop_action(
                        f"Note successfully archived to Vault at: '{saved_path}'"
                    )
                except Exception as e:
                    self.ui_logger.error(
                        f"Failed to save note to Obsidian Vault: {e}"
                    )

            return markdown_content