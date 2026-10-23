import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Central configuration class for the coding agent system."""

    # Multi-provider API keys
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Paths & execution settings
    OBSIDIAN_VAULT_PATH: Path = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./mock_vault"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))

    # Role-specific primary models (OpenRouter)
    # Planner: Fast, highly structured output, context aware
    # Coder: High coding benchmark (DeepSeek V3), cost-effective
    # Tester: Fast, strict JSON/tool output
    AGENT_MODEL_CONFIG: Dict[str, str] = {
        "planner": os.getenv("PLANNER_MODEL", "openrouter/openai/gpt-4o-mini"),
        "coder": os.getenv("CODER_MODEL", "openrouter/deepseek/deepseek-chat"),
        "tester": os.getenv("TESTER_MODEL", "openrouter/openai/gpt-4o-mini"),
    }

    # Default model pipeline for fallbacks (OpenRouter -> Groq -> Gemini)
    DEFAULT_MODEL_FALLBACKS: List[str] = [
        "openrouter/openai/gpt-4o-mini",
        "openrouter/deepseek/deepseek-chat",
        "openrouter/meta-llama/llama-3.3-70b-instruct",
        "groq/openai/gpt-oss-120b",
        "groq/openai/gpt-oss-20b",
    ]

    @classmethod
    def get_model_fallbacks(cls, custom_list: Optional[List[str]] = None) -> List[str]:
        """Return configured model fallbacks, preserving properly formatted model identifiers."""
        models = custom_list
        if models is None:
            override = os.getenv("MODEL_FALLBACK_LIST", "")
            models = override.split(",") if override else cls.DEFAULT_MODEL_FALLBACKS

        return [
            normalized
            for model in models
            if model and (normalized := cls.normalize_model_name(model))
        ]

    @staticmethod
    def normalize_model_name(model_name: str) -> str:
        """Normalize common model aliases to standard provider prefixes."""
        cleaned = model_name.strip()
        if not cleaned:
            return ""

        aliases = {
            "gpt-4o-mini": "openrouter/openai/gpt-4o-mini",
            "gpt-4o": "openrouter/openai/gpt-4o",
            "deepseek-chat": "openrouter/deepseek/deepseek-chat",
            "llama-3.3-70b": "openrouter/meta-llama/llama-3.3-70b-instruct",
        }
        return aliases.get(cleaned.lower(), cleaned)

    @classmethod
    def has_any_api_key(cls) -> bool:
        """Return whether at least one provider credential is configured."""
        return bool(
            cls.OPENROUTER_API_KEY
            or cls.GROQ_API_KEY
            or cls.GEMINI_API_KEY
            or cls.OPENAI_API_KEY
        )

    @classmethod
    def validate(cls) -> None:
        """Validate that at least one AI provider API key is present."""
        if not cls.has_any_api_key():
            raise ValueError(
                "No API keys found. Please set at least OPENROUTER_API_KEY or GROQ_API_KEY in your .env file."
            )