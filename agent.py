import json
import re
import time
from typing import Any

import litellm
from config import Config
from google import genai
from google.genai import types
from google.genai.errors import APIError
from tools.code_editor import CodeEditor

# Suppress unnecessary debug logs from LiteLLM
litellm.suppress_debug_info = True


class CodingAgent:
    """Agent class interfacing with Gemini, Groq, and OpenRouter for multi-file code generation,

    tool execution (Function Calling), and multi-provider fallback handling.
    """

    def __init__(
        self,
        primary_model: str = "gemini-3.6-flash",
        workspace_root: str | None = None,
    ):
        """Initialize AI clients, CodeEditor tools, and provider fallback pipeline."""
        # 1. Initialize Gemini SDK Client (used for native Gemini models)
        self.gemini_client = None
        if Config.GEMINI_API_KEY:
            self.gemini_client = genai.Client(api_key=Config.GEMINI_API_KEY)

        # 2. Initialize CodeEditor tool workspace
        self.code_editor = CodeEditor(workspace_root=workspace_root)

        # 3. Define model fallback pipeline (ordered by priority)
        self.model_pipeline = [
            primary_model,  # e.g. "gemini-2.5-flash"
            "gemini-2.5-pro",
            # --- Groq (Current active models) ---
            "groq/llama-3.3-70b-versatile",
            "groq/llama-3.1-8b-instant",
            # --- OpenRouter (Current free models) ---
            "openrouter/meta-llama/llama-3.3-70b-instruct",
            "openrouter/deepseek/deepseek-r1",
        ]

    def _execute_tool_call(self, tool_name: str, args: dict[str, Any]) -> str:
        """Execute local CodeEditor tool based on LLM function request."""
        try:
            if tool_name == "write_file":
                return self.code_editor.write_file(
                    args["relative_path"], args["content"]
                )
            elif tool_name == "read_file":
                return self.code_editor.read_file(args["relative_path"])
            elif tool_name == "make_directory":
                return self.code_editor.make_directory(args["relative_path"])
            elif tool_name == "list_files":
                return self.code_editor.list_files()
            else:
                return f"Error: Unknown tool function '{tool_name}'"
        except Exception as e:
            return f"Error executing tool '{tool_name}': {e!s}"

    def _call_gemini_native(
        self,
        model: str,
        prompt: str,
        system_instruction: str,
        tools: list[Any] | None = None,
    ) -> str:
        """Call Gemini API natively using official google-genai SDK."""
        if not self.gemini_client:
            raise ValueError("GEMINI_API_KEY is not set.")

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
            tools=tools if tools else None,
        )
        response = self.gemini_client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        return response.text or "Task executed successfully."

    def _call_litellm_provider(
        self, model: str, prompt: str, system_instruction: str, has_tools: bool = False
    ) -> str:
        """Call third-party providers (Groq, OpenRouter) via LiteLLM."""
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ]

        litellm_tools = None
        if has_tools:
            litellm_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "description": "Write or overwrite content to a file inside workspace.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "relative_path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["relative_path", "content"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read file content from workspace.",
                        "parameters": {
                            "type": "object",
                            "properties": {"relative_path": {"type": "string"}},
                            "required": ["relative_path"],
                        },
                    },
                },
            ]

        # LiteLLM Multi-turn tool execution loop
        while True:
            response = litellm.completion(
                model=model,
                messages=messages,
                tools=litellm_tools,
                tool_choice="auto" if litellm_tools else None,
                temperature=0.2,
            )
            response_msg = response.choices[0].message
            tool_calls = getattr(response_msg, "tool_calls", None)

            messages.append(response_msg)

            if not tool_calls:
                return response_msg.content or "Task completed."

            for tool_call in tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                print(f"🔧 [Fallback Tool Call: {model}] -> `{func_name}`")
                tool_result = self._execute_tool_call(func_name, func_args)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": str(tool_result),
                    }
                )

    def _execute_with_fallback(
        self,
        prompt: str,
        system_instruction: str,
        tools: list[Any] | None = None,
    ) -> str:
        """Try models in sequence (Gemini Native -> Groq -> OpenRouter) upon failures or throttling."""
        last_error = None

        for model in self.model_pipeline:
            try:
                if model.startswith("gemini"):
                    # 1. Native Gemini SDK Call
                    return self._call_gemini_native(
                        model=model,
                        prompt=prompt,
                        system_instruction=system_instruction,
                        tools=tools,
                    )
                else:
                    # 2. LiteLLM Proxy Call (Groq / OpenRouter)
                    print(f"📡 Provider Switch: Calling [{model}] via LiteLLM...")
                    return self._call_litellm_provider(
                        model=model,
                        prompt=prompt,
                        system_instruction=system_instruction,
                        has_tools=bool(tools),
                    )

            except (APIError, Exception) as e:
                err_msg = str(e)
                print(
                    f"⚠️ Model/Provider [{model}] unavailable or failed ({err_msg[:100]}...). Switching fallback..."
                )
                last_error = e
                time.sleep(2)
                continue

        raise RuntimeError(
            f"❌ All AI Providers (Gemini, Groq, OpenRouter) failed. Last Error: {last_error}"
        )

    def suggest_project_folder(self, task_instruction: str) -> str:
        """Infer clean project directory name using task instruction."""
        prompt = (
            f"Based on this user task, suggest ONLY a single valid folder name using snake_case or kebab-case (e.g. 'calculator_app'). "
            f"Do not include any extension, markdown formatting, quotes, or explanations.\nTask: {task_instruction}"
        )
        system_instruction = "You are a software architect that determines clean project directory naming."

        try:
            raw_name = self._execute_with_fallback(prompt, system_instruction).strip()
            clean_name = re.sub(r"[```'\"/\s]", "", raw_name)
            return clean_name if clean_name else "generated_project"
        except Exception:
            return "generated_project"

    def suggest_target_file(self, task_instruction: str) -> str:
        """Infer suitable target script filename using task instruction."""
        prompt = (
            f"Based on this user task, suggest ONLY a single valid filename with extension (e.g. 'calculator.py'). "
            f"Do not include any markdown formatting, quotes, or explanations.\nTask: {task_instruction}"
        )
        system_instruction = (
            "You are a software architect that determines clean project file naming."
        )

        try:
            raw_name = self._execute_with_fallback(prompt, system_instruction).strip()
            clean_name = re.sub(r"[```'\"\s]", "", raw_name)
            return clean_name if clean_name else "generated_script.py"
        except Exception:
            return "generated_script.py"

    def generate_code_solution(
        self,
        task_instruction: str,
        target_file_path: str | None = None,
        file_content: str | None = None,
        obsidian_rules: str | None = None,
        error_context: str | None = None,
    ) -> str:
        """Generate code solution with tools and multi-provider self-correction support."""
        prompt_parts = [
            "### Project Development Rules (from Obsidian)",
            obsidian_rules if obsidian_rules else "No specific rules provided.",
        ]

        workspace_summary = self.code_editor.list_files()
        prompt_parts.extend(
            [
                "\n### Existing Workspace Directory Structure",
                "```text",
                workspace_summary if workspace_summary else "Workspace is empty.",
                "```",
            ]
        )

        if target_file_path and file_content:
            prompt_parts.extend(
                [
                    "\n### Primary Target File Context",
                    f"Path: `{target_file_path}`",
                    "```",
                    file_content,
                    "```",
                ]
            )

        prompt_parts.extend(
            [
                "\n### User Task / Instruction",
                task_instruction,
            ]
        )

        if error_context:
            prompt_parts.extend(
                [
                    "\n### ⚠️ Previous Terminal Execution Error (Self-Correction)",
                    "The previous code execution resulted in an error. Fix the issue using tool calls:",
                    "```text",
                    error_context.strip(),
                    "```",
                ]
            )

        full_prompt = "\n".join(prompt_parts)

        system_instruction = (
            "You are an expert autonomous software development assistant.\n"
            "You have direct access to workspace tools to create, read, and manage files:\n"
            "- write_file: Create or update files. Parent directories are created automatically.\n"
            "- read_file: Read content of any workspace file.\n"
            "- make_directory: Explicitly create folder hierarchies.\n"
            "- list_files: Scan workspace files.\n\n"
            "CRITICAL INSTRUCTION: Whenever you generate or update code, ALWAYS execute `write_file` tool calls.\n"
            "You can make multiple `write_file` tool calls in a single turn to build complete multi-file project architectures."
        )

        tools_list = [
            self.code_editor.write_file,
            self.code_editor.read_file,
            self.code_editor.make_directory,
            self.code_editor.list_files,
        ]

        return self._execute_with_fallback(
            full_prompt, system_instruction, tools=tools_list
        )


# --- Standalone Agent Test ---
if __name__ == "__main__":
    print("🤖 Testing Multi-Provider CodingAgent...")
    try:
        agent = CodingAgent(workspace_root="./test_workspace")

        inferred_folder = agent.suggest_project_folder(
            "Build a simple REST API with FastAPI"
        )
        print(f"✅ Project Folder Suggestion: `{inferred_folder}`")

        inferred_file = agent.suggest_target_file(
            "Build a simple REST API with FastAPI"
        )
        print(f"✅ Target File Suggestion: `{inferred_file}`")

    except Exception as err:
        print(f"❌ Test failed: {err}")
