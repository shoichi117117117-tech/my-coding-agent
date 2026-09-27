import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Central configuration class for the coding agent."""

    # Multi-Provider API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GITHUB_API_KEY: str = os.getenv("GITHUB_API_KEY", "")

    # Paths & Execution Settings
    OBSIDIAN_VAULT_PATH: Path = Path(
        os.getenv("OBSIDIAN_VAULT_PATH", "./mock_vault")
    )
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))

    @classmethod
    def validate(cls) -> None:
        """Validate that at least one AI provider API key is present."""
        if not (cls.GEMINI_API_KEY or cls.GROQ_API_KEY or cls.OPENROUTER_API_KEY or cls.OPENAI_API_KEY or cls.GITHUB_API_KEY):
            raise ValueError(
                "No API keys found. Please set at least one of "
                "GEMINI_API_KEY, GROQ_API_KEY, or OPENROUTER_API_KEY, or OPENAI_API_KEY or GITHUB_API_KEYin your .env file."
            )


# Run validation on module import
Config.validate()