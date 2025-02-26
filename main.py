import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from huggingface_hub import InferenceClient

hf_api_key = os.environ.get("HF_API_KEY")

client = InferenceClient(
	provider="novita",
	api_key=hf_api_key
)

messages = [
	{
		"role": "user",
		"content": "How can I learn English easily?"
	}
]

stream = client.chat.completions.create(
	model="meta-llama/Llama-3.1-8B-Instruct", 
	messages=messages, 
	max_tokens=500,
	stream=True
)

for chunk in stream:
    print(chunk.choices[0].delta.content, end="")