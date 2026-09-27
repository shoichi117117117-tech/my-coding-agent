import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
print(f"GROQ_API_KEY present: {bool(groq_api_key)}")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_api_key,
)

try:
    print("Fetching available models from Groq...")
    models = client.models.list()
    
    print("\n--- Available Groq Models ---")
    model_ids = [m.id for m in models.data]
    for m_id in sorted(model_ids):
        print(f"・{m_id}")
    print("------------------------------\n")

    # リストの先頭にあるモデルでテストリクエストを送信
    target_model = model_ids[0] if model_ids else "llama-3.1-8b-instant"
    print(f"Testing with model: [{target_model}]...")

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! Reply with 'Groq connection successful!'"},
        ],
        model=target_model,
    )
    print("🎯 Success!")
    print("Response:", response.choices[0].message.content)

except Exception as e:
    print("❌ Failed with error:")
    print(type(e), e)