from pathlib import Path
from agents.tester import TesterAgent
from tools.obsidian import ObsidianAdapter

# 1. Provide explicit path to your actual Obsidian Vault
vault_path = Path.home() / "Obsidian" / "MyVault"  # Replace with real path

# 2. Instantiate adapter and agent explicitly
adapter = ObsidianAdapter(vault_path=vault_path)
agent = TesterAgent()
agent.obsidian_adapter = adapter  # Attach adapter explicitly

# 3. Check if adapter is recognized before execution
print(f"Obsidian Adapter Status: {agent.obsidian_adapter}")

# 4. Execute generation and saving
result_md = agent.generate_obsidian_troubleshooting_note(
    error_title="Debug Obsidian Vault Persistence",
    error_log="FileNotFoundError: Vault path missing",
    solution_summary="Explicitly attached ObsidianAdapter instance to agent.",
    save_to_vault=True,
)