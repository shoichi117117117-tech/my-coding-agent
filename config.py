import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Central configuration class for the coding agent."""

    # Multi-provider API keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GITHUB_API_KEY: str = os.getenv("GITHUB_API_KEY", "")

    # GitHub Models API Base Endpoint (Azure Inference API)
    GITHUB_MODELS_BASE_URL: str = "https://models.inference.ai.azure.com"

    # Paths & execution settings
    OBSIDIAN_VAULT_PATH: Path = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./mock_vault"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))

    # LiteLLM model identifiers for GitHub Models and fallback providers.
    DEFAULT_MODEL_FALLBACKS: List[str] = [
        "groq/openai/gpt-oss-120b",
        "groq/openai/gpt-oss-20b",
        "groq/qwen/qwen3.8-27b",
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
            "gpt-4o-mini": "github/gpt-4o-mini",
            "gpt-4o": "github/gpt-4o",
            "llama-3.3-70b": "github/meta-llama-3.3-70b-instruct",
        }
        return aliases.get(cleaned.lower(), cleaned)

    @classmethod
    def has_any_api_key(cls) -> bool:
        """Return whether at least one provider credential is configured."""
        return bool(
            cls.GEMINI_API_KEY
            or cls.GROQ_API_KEY
            or cls.OPENROUTER_API_KEY
            or cls.OPENAI_API_KEY
            or cls.GITHUB_API_KEY
        )

    @classmethod
    def validate(cls) -> None:
        """Validate that at least one AI provider API key is present."""
        if not cls.has_any_api_key():
            raise ValueError(
                "No API keys found. Please set at least one of "
                "GEMINI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY, OPENAI_API_KEY, or GITHUB_API_KEY in your .env file."
            )