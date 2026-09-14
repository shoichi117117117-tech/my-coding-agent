import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Central configuration class for the coding agent."""

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OBSIDIAN_VAULT_PATH: Path = Path(
        os.getenv("OBSIDIAN_VAULT_PATH", "./mock_vault")
    )
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))

    @classmethod
    def validate(cls) -> None:
        """Validate that critical environment variables are present."""
        if not cls.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is missing. Please set it in your .env file."
            )


# Run validation on module import
Config.validate()