import os
from datetime import datetime

# Adjust the import path according to your project structure
from my_coding_agent.agents.tester import TesterAgent
from my_coding_agent.config import Config

# 1. Initialize the agent
tester = TesterAgent(
        obsidian_vault_path=str(Config.OBSIDIAN_VAULT_PATH),
    )

# 2. Generate and save a real troubleshooting note to Obsidian Vault
result_md = tester.generate_obsidian_troubleshooting_note(
    error_title="PyYAML Format Verification Test",
    error_log="ScannerError: mapping values are not allowed here in '<string>', line 3, column 9",
    solution_summary="Constructed YAML frontmatter using raw multiline string formatted explicitly without textwrap.dedent.",
    save_to_vault=True,  # Save directly to Obsidian Vault
)

print("Generated Markdown Output:\n", result_md)
