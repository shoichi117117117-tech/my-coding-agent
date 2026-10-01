"""Unit tests for Pydantic-driven query extraction in BaseAgent and relevance-scored Obsidian search."""

from pathlib import Path
from unittest.mock import MagicMock
import pytest
from pydantic import BaseModel

from agents.base import BaseAgent, SearchQueryExtraction
from tools.obsidian import ObsidianAdapter


class MockAgent(BaseAgent):
    """Mock agent inheriting from BaseAgent to test structured query extraction behavior."""

    def __init__(self, mocked_keywords: list[str]) -> None:
        """Initialize mock agent with predefined keyword responses."""
        self.mocked_keywords = mocked_keywords
        self.logger = MagicMock()  # Mock Logger to avoid file/UI dependencies

    def call_llm_structured(
        self, messages: list, response_format: type[BaseModel]
    ) -> SearchQueryExtraction:
        """Simulate LLM response returning SearchQueryExtraction instance directly."""
        return SearchQueryExtraction(keywords=self.mocked_keywords)


@pytest.fixture
def temp_vault(tmp_path: Path) -> Path:
    """Set up a temporary Obsidian Vault structure with sample troubleshooting notes."""
    vault_dir = tmp_path / "ObsidianVault"
    trouble_dir = vault_dir / "30_Resources" / "Troubleshooting"
    trouble_dir.mkdir(parents=True)

    # High-relevance note (Title + Tags match)
    note_1 = trouble_dir / "Pydantic_ValidationError.md"
    note_1.write_text(
        "---\n"
        "type: troubleshooting\n"
        "tags:\n"
        "  - pydantic\n"
        "  - validation\n"
        "---\n\n"
        "# [Issue] Pydantic ValidationError Fix\n\n"
        "Use model_validate_json to parse structured outputs safely.",
        encoding="utf-8",
    )

    # Low-relevance note (Body match only)
    note_2 = trouble_dir / "General_Python_Tips.md"
    note_2.write_text(
        "---\n"
        "type: resource\n"
        "tags:\n"
        "  - python\n"
        "---\n\n"
        "# [Guide] Python Best Practices\n\n"
        "When handling errors, log tracebacks clearly. pydantic is useful here.",
        encoding="utf-8",
    )

    return vault_dir


def test_short_query_bypass() -> None:
    """Verify that short queries bypass LLM structured extraction within BaseAgent."""
    agent = MockAgent(mocked_keywords=["unused"])
    
    # Short query under 40 chars and <= 3 words should bypass call_llm_structured
    keywords = agent._extract_search_keywords("pydantic validation")
    assert keywords == ["pydantic", "validation"]


def test_noisy_query_extraction() -> None:
    """Verify that noisy long queries are converted to clean keywords using SearchQueryExtraction."""
    agent = MockAgent(mocked_keywords=["pydantic", "validationerror"])

    noisy_stacktrace = (
        'Traceback (most recent call last):\n'
        '  File "/usr/local/lib/python3.11/site-packages/pydantic/main.py", line 341, in __init__\n'
        'pydantic.errors.ValidationError: 1 validation error for TroubleshootingAnalysis'
    )

    keywords = agent._extract_search_keywords(noisy_stacktrace)
    assert keywords == ["pydantic", "validationerror"]


def test_relevance_scoring_priority(temp_vault: Path) -> None:
    """Verify that ObsidianAdapter ranks notes with title/tag matches higher in search results."""
    adapter = ObsidianAdapter(vault_path=temp_vault)
    
    # Pass clean keywords list extracted by BaseAgent
    content, sources = adapter.search_with_sources(query=["pydantic"])

    assert len(sources) == 2
    # The note with title and tag matches must be ranked first
    assert "Pydantic_ValidationError.md" in sources[0]
    assert "Relevance Score:" in content