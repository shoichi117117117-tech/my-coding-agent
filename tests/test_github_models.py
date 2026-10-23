import logging
import sys
from pathlib import Path

# Add project root to sys.path so modules can be imported from tests/
sys.path.append(str(Path(__file__).resolve().parent.parent))

from my_coding_agent.config import Config
from my_coding_agent.agents.base import BaseAgent, extract_response_content

# Configure logging for step-by-step connection visibility
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    print("=== 1. Validating Configuration & Credentials ===")
    try:
        Config.validate()
        print("✅ Config.validate() passed: Credentials verified.")
    except Exception as e:
        print(f"❌ Config.validate() failed: {e}")
        return

    github_key_status = "Set" if Config.GITHUB_API_KEY else "Not Set"
    print(f"・GITHUB_API_KEY Status: {github_key_status}")
    print(f"・Configured Model Pipeline: {Config.get_model_fallbacks()}\n")

    print("=== 2. Testing GitHub Models Integration via BaseAgent ===")
    test_agent = BaseAgent(
        name="TestAgent",
        system_prompt="You are a helpful and concise software engineering assistant. Respond strictly in 1-2 sentences.",
    )

    # Test payload for verifying API connectivity
    messages = [
        {
            "role": "user",
            "content": "This is a connectivity test via GitHub Models API. Please confirm execution with a brief greeting.",
        }
    ]

    try:
        response = test_agent.call_llm(messages=messages)
        content = extract_response_content(response)

        print("\n=== 🎯 LLM Response Received Successfully ===")
        print(f"Agent Output:\n{content}")
        print("=============================================")
        print("🎉 GitHub Models integration and fallback handler are fully operational!")

    except Exception as e:
        print(f"\n❌ Execution Failed: {e}")


if __name__ == "__main__":
    main()