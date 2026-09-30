"""Base Agent module handling LLM call abstractions, multi-provider fallback routing, and structured Pydantic outputs."""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import litellm
from google import genai
from google.genai import types
from pydantic import BaseModel

from config import Config
from utils.ui import AgentLogger

# Configure structured root logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Python default logger configuration
logger = logging.getLogger("BaseAgent")
litellm.suppress_debug_info = True

T = TypeVar("T", bound=BaseModel)


def extract_response_content(response: Any) -> str:
    """Extract string content cleanly from various LLM response objects across different SDKs."""
    if isinstance(response, str):
        return response

    if hasattr(response, "choices") and response.choices:
        choice = response.choices[0]
        if hasattr(choice, "message"):
            msg = choice.message
            if hasattr(msg, "content") and msg.content is not None:
                return msg.content

    if hasattr(response, "content") and response.content is not None:
        return response.content

    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices and isinstance(choices, list):
            message = choices[0].get("message", {})
            if isinstance(message, dict) and "content" in message:
                return message.get("content") or ""

    return str(response)


class BaseAgent:
    """Base Agent providing multi-provider model routing, fallback execution pipelines, and structured output parsing."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        role: Optional[str] = None,
        model_fallback_list: Optional[List[str]] = None,
    ) -> None:
        """Initialize base agent configurations and model execution pipelines."""
        self.name = name
        self.system_prompt = system_prompt
        self.role = role.lower() if role else name.lower()

        # UI logger initialization per agent
        self.logger = AgentLogger(agent_name=self.name)

        # Native Gemini client initialization
        self.gemini_client: Optional[genai.Client] = None
        if getattr(Config, "GEMINI_API_KEY", None):
            self.gemini_client = genai.Client(api_key=Config.GEMINI_API_KEY)

        # Build execution pipeline prioritizing assigned role model
        pipeline: List[str] = []
        if self.role in Config.AGENT_MODEL_CONFIG:
            primary_model = Config.AGENT_MODEL_CONFIG[self.role]
            pipeline.append(primary_model)

        default_fallbacks = Config.get_model_fallbacks(model_fallback_list)
        for model in default_fallbacks:
            if model not in pipeline:
                pipeline.append(model)

        self.model_pipeline = pipeline
        logger.info(f"[{self.name}] Initialized model pipeline: {self.model_pipeline}")

    def _call_gemini_native(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None,
    ) -> Any:
        """Execute request using Google GenAI SDK natively, handling both JSON and Pydantic schema formats."""
        if not self.gemini_client:
            raise ValueError("GEMINI_API_KEY is not configured.")

        user_content = "\n".join([m.get("content", "") for m in messages if m.get("content")])
        
        # Determine MIME type or structured schema for Gemini SDK
        is_structured = response_format is not None
        mime_type = "application/json" if is_structured else "text/plain"

        config_kwargs: Dict[str, Any] = {
            "system_instruction": self.system_prompt,
            "temperature": 0.2,
            "response_mime_type": mime_type,
        }

        # Inject Pydantic schema into response_schema if provided
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            config_kwargs["response_schema"] = response_format

        config = types.GenerateContentConfig(**config_kwargs)

        response = self.gemini_client.models.generate_content(
            model=model,
            contents=user_content,
            config=config,
        )

        class MockChoice:
            def __init__(self, text: str) -> None:
                self.message = type("Message", (), {"content": text, "tool_calls": None})()

        class MockResponse:
            def __init__(self, text: str) -> None:
                self.choices = [MockChoice(text)]

        return MockResponse(response.text or "Task executed successfully.")

    def _call_litellm_provider(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None,
    ) -> Any:
        """Execute request using LiteLLM unified completion endpoint supporting Pydantic response formats."""
        full_messages = [{"role": "system", "content": self.system_prompt}] + messages

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": full_messages,
            "temperature": 0.2,
        }

        if model.startswith("openrouter/") and getattr(Config, "OPENROUTER_API_KEY", None):
            kwargs["api_key"] = Config.OPENROUTER_API_KEY

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        if response_format:
            kwargs["response_format"] = response_format

        return litellm.completion(**kwargs)

    def call_llm(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None,
    ) -> Any:
        """Execute completion calls sequentially across model pipeline with retry backoff."""
        last_error: Optional[Exception] = None

        for idx, model in enumerate(self.model_pipeline):
            try:
                if "openrouter" in model:
                    provider = "openrouter"
                elif "gemini" in model:
                    provider = "google"
                else:
                    provider = "groq"

                # Log request metadata for CLI UI output
                self.logger.log_llm_request(model_name=model, provider=provider)

                if model.startswith("gemini/") or model.startswith("gemini-"):
                    return self._call_gemini_native(
                        model=model,
                        messages=messages,
                        tools=tools,
                        response_format=response_format,
                    )
                return self._call_litellm_provider(
                    model=model,
                    messages=messages,
                    tools=tools,
                    response_format=response_format,
                )

            except Exception as e:
                err_msg = str(e)
                self.logger.warning(
                    f"Model [{model}] failed ({err_msg[:100]}...). Initiating fallback..."
                )
                last_error = e

                wait_time = min(2 ** (idx + 1), 8)
                time.sleep(wait_time)
                continue

        raise RuntimeError(
            f"❌ All configured AI providers failed for Agent [{self.name}]. Last Error: {last_error}"
        )

    def call_llm_structured(
        self,
        messages: List[Dict[str, Any]],
        response_format: Type[T],
    ) -> T:
        """Execute LLM call enforcing structured outputs and returning a validated Pydantic model instance."""
        raw_response = self.call_llm(
            messages=messages,
            response_format=response_format,
        )

        content = extract_response_content(raw_response).strip()

        # Strip surrounding markdown json block fences if returned by provider
        if content.startswith("```"):
            content = re.sub(
                r"^```(?:json)?\s*([\s\S]*?)\s*```$",
                r"\1",
                content,
                flags=re.IGNORECASE,
            ).strip()

        # Parse raw JSON string into target Pydantic class instance
        try:
            return response_format.model_validate_json(content)
        except Exception as parse_error:
            # Fallback handling for dict parsing if raw JSON conversion fails
            try:
                data = json.loads(content)
                return response_format.model_validate(data)
            except Exception:
                raise ValueError(
                    f"Failed to parse LLM structured output into Pydantic model {response_format.__name__}.\n"
                    f"Raw Response: {content}\n"
                    f"Parse Error: {parse_error}"
                )