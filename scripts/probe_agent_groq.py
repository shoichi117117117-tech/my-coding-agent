import logging
import sys
from pathlib import Path

# Add project root directory to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from my_coding_agent.config import Config
from my_coding_agent.agents.base import BaseAgent, extract_response_content

# Configure logging format for test execution
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("TestGroqAgent")


def test_agent_with_groq():
    print("\n=== 1. Validating Configuration & Groq Model Pipeline ===")
    
    # Target Groq model fallbacks verified in active model listing
    groq_models = [
        "groq/openai/gpt-oss-120b",
        "groq/openai/gpt-oss-20b",
        "groq/qwen/qwen3.8-27b",
    ]
    
    print(f"・Configured Model Pipeline: {groq_models}")
    Config.validate()
    print("✅ Credentials / API Key verification passed.\n")

    print("=== 2. Testing BaseAgent Execution via Groq ===")
    
    # Instantiate BaseAgent with Groq pipeline
    agent = BaseAgent(
        name="GroqTesterAgent",
        system_prompt="You are an expert software engineer. Provide concise and accurate answers.",
        model_fallback_list=groq_models,
    )

    # Test prompt payload
    messages = [
        {"role": "user", "content": "Write a single line of Python code to remove duplicates from a list."}
    ]

    try:
        logger.info("Sending request payload to agent...")
        raw_response = agent.call_llm(messages=messages)
        
        # Extract plain string response content
        content = extract_response_content(raw_response)
        
        print("\n" + "=" * 40)
        print("🎯 Test Succeeded! Response from Agent:")
        print("=" * 40)
        print(content)
        print("=" * 40 + "\n")

    except Exception as e:
        print(f"\n❌ Agent execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    test_agent_with_groq()