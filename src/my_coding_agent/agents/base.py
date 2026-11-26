"""Base Agent module handling LLM call abstractions, multi-provider fallback routing, and structured Pydantic outputs."""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import litellm
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from my_coding_agent.config import Config
from my_coding_agent.utils.ui import AgentLogger

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
            if hasattr(msg, "content"):
                return msg.content or ""

    if hasattr(response, "content"):
        return response.content or ""

    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices and isinstance(choices, list):
            message = choices[0].get("message", {})
            if isinstance(message, dict) and "content" in message:
                return message.get("content") or ""

    return str(response)

# --- Pydantic Schema for Query Extraction ---
class SearchQueryExtraction(BaseModel):
    """Pydantic model for extracting clean search keywords from noisy or long raw queries."""

    keywords: List[str] = Field(
        description="Extract 1-4 core technical terms, error names, or specific concepts suitable for search (e.g., ['Pydantic', 'ValidationError', 'yaml'])."
    )

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
            primary_model = Config.normalize_model_name(Config.AGENT_MODEL_CONFIG[self.role])
            if primary_model:
                pipeline.append(primary_model)

        default_fallbacks = Config.get_model_fallbacks(model_fallback_list)
        for model in default_fallbacks:
            if model not in pipeline:
                pipeline.append(model)

        self.model_pipeline = pipeline
        logger.info(f"[{self.name}] Initialized model pipeline: {self.model_pipeline}")

    
    def _extract_search_keywords(self, raw_query: str) -> List[str]:
        """Extract clean technical keywords using the agent's assigned LLM model via structured outputs."""
        # Short clean queries bypass LLM extraction to save latency and token costs
        words = raw_query.strip().split()
        if len(words) <= 3 and len(raw_query) < 40:
            return [w.lower() for w in words if w]

        try:
            prompt = (
                f"Extract key technical search terms from the following raw query or error log.\n"
                f"Remove file paths, line numbers, standard english sentence fluff, and noisy stack trace lines.\n\n"
                f"Raw Query:\n{raw_query}"
            )
            messages = [{"role": "user", "content": prompt}]

            # Call the agent's own structured LLM pipeline
            extracted: SearchQueryExtraction = self.call_llm_structured(
                messages=messages,
                response_format=SearchQueryExtraction,
            )
            return [k.lower().strip() for k in extracted.keywords if k.strip()]
        except Exception as e:
            self.logger.warning(f"Keyword extraction failed, falling back to basic split: {e}")
            return [w.lower() for w in words if len(w) > 2]

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

        if tools:
            raise NotImplementedError(
                "Native Gemini client path does not execute OpenAI-format tool definitions; falling back."
            )

        user_content = "\n".join(
            f"[{m.get('role', 'user').upper()}]: {m.get('content', '')}"
            for m in messages
            if m.get("content")
        )
        
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
                if model.startswith("openrouter/"):
                    provider = "openrouter"
                elif model.startswith("gemini/") or model.startswith("gemini-"):
                    provider = "google"
                elif model.startswith("groq/"):
                    provider = "groq"
                elif model.startswith("openai/") or "/" not in model:
                    provider = "openai"
                else:
                    provider = model.split("/", 1)[0]

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