import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

github_token = os.getenv("GITHUB_API_KEY")

print(f"Token present: {bool(github_token)}")

# GitHub Models API configuration for OpenAI SDK
client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=github_token,
)

try:
    print("Sending request to GitHub Models (gpt-4o-mini)...")
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! Reply with 'Connection successful!'"},
        ],
        model="gpt-4o-mini",
    )
    print("🎯 Success!")
    print("Response:", response.choices[0].message.content)

except Exception as e:
    print("❌ Failed with error:")
    print(type(e), e)