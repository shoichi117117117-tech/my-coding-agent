import logging
import time
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from google.genai.errors import APIError
import litellm

from config import Config

logger = logging.getLogger(__name__)
litellm.suppress_debug_info = True


def extract_response_content(response: Any) -> str:
    """Extract pure string content cleanly from various LLM response objects or providers."""
    if isinstance(response, str):
        return response

    # 1. Standard OpenAI / LiteLLM response structure (choices[0].message.content)
    if hasattr(response, "choices") and response.choices:
        choice = response.choices[0]
        if hasattr(choice, "message"):
            msg = choice.message
            if hasattr(msg, "content") and msg.content is not None:
                return msg.content

    # 2. Direct attribute fallback (.content)
    if hasattr(response, "content") and response.content is not None:
        return response.content

    # 3. Dict style response fallback
    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices and isinstance(choices, list):
            message = choices[0].get("message", {})
            if isinstance(message, dict) and "content" in message:
                return message.get("content") or ""

    # Fallback string representation
    return str(response)


class BaseAgent:
    """Base Agent class handling multi-provider LLM calls (Gemini Native -> Groq -> OpenRouter)
    with automatic failover/fallback logic.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        model_fallback_list: Optional[List[str]] = None,
    ):
        self.name = name
        self.system_prompt = system_prompt

        # Gemini Native Client
        self.gemini_client = None
        if getattr(Config, "GEMINI_API_KEY", None):
            self.gemini_client = genai.Client(api_key=Config.GEMINI_API_KEY)

        # Multi-provider fallback pipeline
        self.model_pipeline = model_fallback_list or [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "groq/llama-3.3-70b-versatile",
            "groq/llama-3.1-8b-instant",
            "openrouter/meta-llama/llama-3.3-70b-instruct",
            "openrouter/deepseek/deepseek-r1",
        ]

    def _call_gemini_native(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if not self.gemini_client:
            raise ValueError("GEMINI_API_KEY is not set.")

        # Convert OpenAI style messages to simple string context for native call
        user_content = "\n".join([m.get("content", "") for m in messages if m.get("content")])

        # Handle JSON mode for Gemini Native Config
        mime_type = "application/json" if response_format and response_format.get("type") == "json_object" else "text/plain"

        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            temperature=0.2,
            response_mime_type=mime_type,
        )

        response = self.gemini_client.models.generate_content(
            model=model,
            contents=user_content,
            config=config,
        )

        # Mocking OpenAI response format for standard interface
        class MockChoice:
            def __init__(self, text):
                self.message = type("Message", (), {"content": text, "tool_calls": None})()

        class MockResponse:
            def __init__(self, text):
                self.choices = [MockChoice(text)]

        return MockResponse(response.text or "Task executed successfully.")

    def _call_litellm_provider(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Any:
        full_messages = [{"role": "system", "content": self.system_prompt}] + messages

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": full_messages,
            "temperature": 0.2,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        if response_format:
            kwargs["response_format"] = response_format

        response = litellm.completion(**kwargs)
        return response

    def call_llm(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Executes LLM call trying models in sequence (Gemini -> Groq -> OpenRouter)."""
        last_error = None

        for model in self.model_pipeline:
            try:
                if model.startswith("gemini"):
                    logger.info(f"[{self.name}] Calling Native Gemini model: {model}")
                    return self._call_gemini_native(
                        model=model,
                        messages=messages,
                        tools=tools,
                        response_format=response_format,
                    )
                else:
                    logger.info(f"[{self.name}] Provider Switch: Calling [{model}] via LiteLLM...")
                    return self._call_litellm_provider(
                        model=model,
                        messages=messages,
                        tools=tools,
                        response_format=response_format,
                    )

            except Exception as e:
                logger.warning(
                    f"[{self.name}] Model/Provider [{model}] failed ({str(e)[:100]}...). Switching fallback..."
                )
                last_error = e
                time.sleep(1)
                continue

        raise RuntimeError(f"❌ All AI Providers failed for Agent [{self.name}]. Last Error: {last_error}")