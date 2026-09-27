import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from agents.base import BaseAgent, extract_response_content
from tools.obsidian import ObsidianAdapter

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """You are the Lead Software Architect and Planner Agent responsible for analyzing user requests, breaking down complex software development tasks into structured sequential steps, diagnosing runtime/terminal errors, and suggesting clean directory architectures and verification strategies.

Current Workspace Root: {workspace_root}
Obsidian Vault Path: {obsidian_vault_path}

Your objectives:
1. Break down high-level user requests or error resolution strategies into actionable sub-tasks for Coder and Tester agents.
2. Determine project language/stack requirements and mandate appropriate configuration manifest creation (e.g., Python: `pyproject.toml`, Node.js: `package.json`, Java: `pom.xml`, Rust: `Cargo.toml`).
3. Output clear, step-by-step execution plans in pure JSON format.
4. Analyze tasks to infer flexible verification strategies (e.g., unit testing, building, direct execution, or static preview).
5. Determine clean directory structures and primary filenames when requested.

Standard Project Layout & Execution Guidelines:
1. Manifest Selection Rules:
   - STEP 1 MUST instruct Coder to write or update valid project configuration manifests for the target language stack.
   - Coder has access to `search_obsidian_vault` to query exact manifest schemas if needed.
   - STEP 2 (if dependencies are required) MUST specify environment synchronization via CLI command (e.g., `uv sync`, `npm install`, `mvn compile`, `cargo check`).

2. Directory Structure & Test Placement:
   - Python Projects: Unit test files MUST ALWAYS be placed inside a `tests/` directory (e.g., `tests/test_main.py`), NEVER flat in the root directory.
   - JavaScript/TypeScript: Source code in `src/`, unit tests in `tests/` or `__tests__/`.

JSON Output Format Requirement for create_plan():
{{
  "steps": [
    {{
      "id": 1,
      "description": "Initialize pyproject.toml with necessary dependencies and pytest configurations.",
      "assigned_to": "coder",
      "target_files": ["pyproject.toml"]
    }},
    {{
      "id": 2,
      "description": "Synchronize environment dependencies.",
      "assigned_to": "coder",
      "command_to_execute": "uv sync"
    }},
    {{
      "id": 3,
      "description": "Implement target feature code and unit test suite inside tests/ directory.",
      "assigned_to": "coder",
      "target_files": ["main.py", "tests/test_main.py"]
    }},
    {{
      "id": 4,
      "description": "Verify implementation via automated test suite.",
      "assigned_to": "tester",
      "command_to_execute": "uv run pytest tests/test_main.py"
    }}
  ]
}}

Strict Rules:
1. "assigned_to" must be either "coder" or "tester".
2. Keep step descriptions clear, concise, and focused on a single responsibility.
3. Include "command_to_execute" when a step requires terminal execution.
4. ALL inner notes, descriptions, and comments must be written in ENGLISH.
5. Output ONLY a valid JSON block.
"""


class PlannerAgent(BaseAgent):
    """Planner Agent that decomposes high-level requests into actionable sub-tasks,
    infers verification strategies, generates rescue plans from terminal errors, and integrates with Obsidian Vault.
    """

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        obsidian_vault_path: Optional[str] = None,
        model_fallback_list: Optional[List[str]] = None,
    ):
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else None
        self.obsidian_vault_path = Path(obsidian_vault_path).resolve() if obsidian_vault_path else None
        self.obsidian_adapter = ObsidianAdapter(str(self.obsidian_vault_path)) if self.obsidian_vault_path else None

        formatted_system_prompt = PLANNER_SYSTEM_PROMPT.format(
            workspace_root=str(self.workspace_root) if self.workspace_root else "Not specified",
            obsidian_vault_path=str(self.obsidian_vault_path) if self.obsidian_vault_path else "Not specified",
        )

        super().__init__(
            name="Planner",
            system_prompt=formatted_system_prompt,
            model_fallback_list=model_fallback_list,
        )

    def _clean_json_response(self, raw_content: str) -> str:
        """Helper method to extract a clean JSON string from raw LLM responses using regex."""
        content = raw_content.strip()
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return content

    def suggest_project_structure(self, task_instruction: str) -> Dict[str, str]:
        """Infers clean project directory name (snake_case) and main target filename based on user task."""
        clean_instruction = re.sub(r"\s+", " ", task_instruction).strip()
        short_task = clean_instruction[:500]

        prompt = (
            f"Analyze this user task and propose a specific, descriptive lower_snake_case folder name "
            f"(1-3 words max, directly reflecting the task, e.g., 'fastapi_fibonacci', 'sales_csv_analyzer', 'email_validator') "
            f"and a primary target filename (e.g., main.py, index.js, App.java).\n\n"
            f"Task: {short_task}\n\n"
            f"CRITICAL RULES:\n"
            f"- DO NOT use generic names like 'my_project', 'generated_project', 'project', or 'app'.\n"
            f"- Folder name MUST directly reflect the user's specific request.\n"
            f"- Output ONLY valid JSON matching this exact structure:\n"
            f'{{"folder_name": "descriptive_name_here", "main_file": "main.py"}}'
        )

        fallback_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            response = self.call_llm(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            raw_content = extract_response_content(response)
            cleaned_str = self._clean_json_response(raw_content)
            parsed = json.loads(cleaned_str)

            raw_folder = parsed.get("folder_name", "")
            raw_main = parsed.get("main_file", "")

            # Convert spaces and hyphens to underscores and clean invalid characters
            folder = re.sub(r"[^a-z0-9_]", "", raw_folder.lower().replace(" ", "_").replace("-", "_")).strip("_")
            main_file = re.sub(r"[^\w\.-]", "", raw_main).strip()

            # Guard against fallback or overly generic folder names
            generic_names = {"generated_project", "my_project", "project", "app", "python_project", "test_project"}
            if not folder or folder in generic_names:
                words = re.findall(r"\b[a-zA-Z]{3,}\b", short_task.lower())
                meaningful_words = [w for w in words if w not in {"create", "build", "make", "using", "python", "with", "test", "file"}]
                if meaningful_words:
                    folder = f"{'_'.join(meaningful_words[:2])}_app"
                else:
                    folder = f"project_{fallback_timestamp}"

            return {
                "folder_name": folder,
                "main_file": main_file if main_file else "main.py",
            }
        except Exception as e:
            logger.warning(f"[Planner] Could not parse structure suggestion ({e}), falling back to timestamped directory.")
            return {"folder_name": f"project_{fallback_timestamp}", "main_file": "main.py"}

    def suggest_project_folder(self, task_instruction: str) -> str:
        """Extract folder name from project structure suggestion."""
        structure = self.suggest_project_structure(task_instruction)
        return structure.get("folder_name", f"project_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    def analyze_verification_strategy(self, task_instruction: str) -> Dict[str, Optional[str]]:
        """Dynamically analyzes the task requirement and determines the appropriate verification strategy."""
        structure = self.suggest_project_structure(task_instruction)
        target_file = structure.get("main_file", "main.py")

        prompt = (
            f"Task Instruction: {task_instruction}\n"
            f"Primary Target File: {target_file}\n\n"
            "Analyze the nature of this project (e.g., Python backend, React/Node.js frontend, Java Maven/Gradle, Rust CLI, or static HTML/CSS).\n"
            "Determine the verification mode and appropriate terminal verification command.\n\n"
            "Modes:\n"
            "- 'unit_test': Requires generating unit tests (e.g., pytest, jest, junit, cargo test). Note: Python test files MUST be placed under `tests/`.\n"
            "- 'build_or_lint': For compiled or bundled projects requiring build validation (e.g., `npm run build`, `mvn compile`).\n"
            "- 'direct_execution': Directly execute script to verify no runtime syntax errors (e.g., `uv run python main.py`, `node index.js`).\n"
            "- 'none': Pure documentation or static assets.\n\n"
            "Respond ONLY in valid JSON:\n"
            "{\n"
            '  "target_file": "main.py",\n'
            '  "test_file": "tests/test_main.py",\n'
            '  "test_command": "uv run pytest tests/test_main.py",\n'
            '  "mode": "unit_test"\n'
            "}"
        )

        try:
            response = self.call_llm(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            raw_content = extract_response_content(response)
            cleaned_str = self._clean_json_response(raw_content)
            parsed = json.loads(cleaned_str)

            return {
                "target_file": parsed.get("target_file", target_file),
                "test_file": parsed.get("test_file"),
                "test_command": parsed.get("test_command"),
                "mode": parsed.get("mode", "direct_execution"),
            }
        except Exception as e:
            logger.warning(f"[Planner] Failed to infer verification strategy ({e}). Using default fallback strategy.")

            stem = Path(target_file).stem
            test_file = f"tests/test_{stem}.py" if target_file.endswith(".py") else None
            test_command = f"uv run pytest {test_file}" if test_file else f"python {target_file}"

            return {
                "target_file": target_file,
                "test_file": test_file,
                "test_command": test_command,
                "mode": "unit_test" if target_file.endswith(".py") else "direct_execution",
            }

    def create_plan(
        self,
        task_instruction: str,
        target_file: str = "main.py",
        domain_rules: str = "",
        error_context: Optional[str] = None,
        attempt: int = 1,
        auto_fetch_vault_rules: bool = True,
    ) -> Union[List[Dict[str, Any]], str]:
        """Creates a structured JSON execution or rescue plan based on instructions, domain rules, and error context."""
        if auto_fetch_vault_rules and self.obsidian_adapter:
            vault_rules = self.obsidian_adapter.read_rules(task_instruction=task_instruction)
            active_rules = f"{domain_rules}\n\n{vault_rules}".strip()
        else:
            active_rules = domain_rules.strip() if domain_rules else "No specific rules provided."

        if error_context:
            formatted_user_message = (
                f"【ATTEMPT #{attempt} FAILED】\n"
                f"Target File: {target_file}\n"
                f"Target Goal: {task_instruction}\n\n"
                f"【Terminal / Test Error Output & RCA】\n"
                f"```text\n{error_context}\n```\n\n"
                f"【Active Development Rules】\n{active_rules}\n\n"
                "Please perform a Root Cause Analysis and output a JSON execution plan (Rescue Plan) for Coder/Tester."
            )
            logger.info(f"[Planner] Generating JSON Rescue Plan for attempt #{attempt}...")
        else:
            formatted_user_message = (
                f"【Target File】: {target_file}\n"
                f"【Active Development Rules】\n{active_rules}\n\n"
                f"【User Request】\n{task_instruction}\n\n"
                "Please output a structured JSON execution plan for this task."
            )
            logger.info("[Planner] Generating task decomposition plan...")

        messages = [{"role": "user", "content": formatted_user_message}]

        response = self.call_llm(
            messages=messages,
            response_format={"type": "json_object"},
        )
        raw_content = extract_response_content(response)

        cleaned_json_str = self._clean_json_response(raw_content)

        try:
            parsed_plan = json.loads(cleaned_json_str)
            steps = parsed_plan.get("steps", [])
            logger.info(f"[Planner] Plan successfully generated with {len(steps)} steps.")
            return json.dumps(parsed_plan, indent=2, ensure_ascii=False)
        except json.JSONDecodeError as e:
            logger.error(f"[Planner] Failed to parse JSON plan output: {e}\nRaw output:\n{raw_content}")
            return cleaned_json_str