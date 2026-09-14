import os
import re
from typing import Optional, List
from google import genai
from google.genai import types
from google.genai.errors import APIError
from config import Config


class CodingAgent:
    """Agent class interfacing with Gemini API for code generation and self-correction."""

    def __init__(
        self,
        primary_model: str = "gemini-3.6-flash",
        fallback_models: Optional[List[str]] = None,
    ):
        """
        Initialize the Gemini Client with primary and fallback models.
        """
        if not Config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not configured.")

        self.client = genai.Client(api_key=Config.GEMINI_API_KEY)
        self.primary_model = primary_model
        self.fallback_models = fallback_models or ["gemini-3.5-flash", "gemini-3.6-pro"]

    def _call_gemini_model(
        self, model: str, prompt: str, system_instruction: str
    ) -> str:
        """Core private method to handle single Gemini API calls."""
        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
            ),
        )
        return response.text

    def _execute_with_fallback(self, prompt: str, system_instruction: str) -> str:
        """Try primary model first, fallback to backup models if throttled/busy."""
        models_to_try = [self.primary_model] + self.fallback_models

        for model in models_to_try:
            try:
                return self._call_gemini_model(model, prompt, system_instruction)
            except APIError as e:
                if getattr(e, "code", None) in [503, 429] or "busy" in str(e).lower():
                    print(f"⚠️ Model {model} is busy/throttled. Trying fallback model...")
                    continue
                raise e
            except Exception as e:
                print(f"⚠️ Error with {model}: {e}. Trying fallback...")
                continue

        raise RuntimeError("❌ All primary and fallback models failed to respond.")

    def suggest_target_file(self, task_instruction: str) -> str:
        """Infer a suitable filename based on user task instruction."""
        prompt = (
            f"Based on this user task, suggest ONLY a single valid filename with extension (e.g. 'calculator.py'). "
            f"Do not include any markdown formatting, quotes, or explanations.\nTask: {task_instruction}"
        )
        system_instruction = "You are a software architect that determines clean project file naming."

        try:
            raw_name = self._execute_with_fallback(prompt, system_instruction).strip()
            clean_name = re.sub(r"[```'\"\s]", "", raw_name)
            return clean_name if clean_name else "generated_script.py"
        except Exception:
            return "generated_script.py"

    def generate_code_solution(
        self,
        task_instruction: str,
        target_file_path: str,
        file_content: str,
        obsidian_rules: str,
        error_context: Optional[str] = None,
    ) -> str:
        """Generate or modify code using centralized Gemini caller with fallback support."""
        prompt_parts = [
            "### Project Development Rules (from Obsidian)",
            obsidian_rules if obsidian_rules else "No specific rules provided.",
            "\n### Target File",
            f"Path: `{target_file_path}`",
            "```",
            file_content if file_content else "# Empty or new file",
            "```",
            "\n### User Task / Instruction",
            task_instruction,
        ]

        if error_context:
            prompt_parts.extend([
                "\n### ⚠️ Previous Terminal Execution Error (Self-Correction)",
                "The previous code update caused an error when executed. Please fix the issue:",
                "```text",
                error_context.strip(),
                "```",
            ])

        prompt_parts.extend([
            "\n### Instructions for Output",
            "1. Analyze the requirements and existing file content carefully.",
            "2. Provide a clear solution that adheres to the development rules.",
            "3. Return the complete updated file inside markdown code blocks (e.g., ```python ... ```).",
        ])

        full_prompt = "\n".join(prompt_parts)
        system_instruction = (
            "You are an expert autonomous software development assistant. "
            "Your goal is to write clean, maintainable, error-free code that strictly follows "
            "the provided project rules and fixes any reported terminal errors."
        )

        return self._execute_with_fallback(full_prompt, system_instruction)


# --- Quick Standalone Test ---
if __name__ == "__main__":
    print("🤖 Testing refactored CodingAgent...")
    try:
        agent = CodingAgent()
        inferred_file = agent.suggest_target_file("Create a basic calculator module")
        print(f"✅ Target file suggestion test passed: `{inferred_file}`")
    except Exception as err:
        print(f"❌ Test failed: {err}")