# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv

# load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation"
# )
# model = ChatHuggingFace(llm=llm)

# result = model.invoke("What is the capital of India")

# print(result.content)

import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    print("Error: HUGGINGFACEHUB_API_TOKEN not found in environment variables.")
    print("Please set it in your .env file or directly in the script.")
else:
    try:
        # Try with TinyLlama first
        client = InferenceClient(token=HF_TOKEN)
        # For text-generation, you use .text_generation()
        response = client.text_generation(
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            prompt="What is the capital of India?",
            max_new_tokens=50,
        )
        print(f"TinyLlama direct client response: {response}")

    except Exception as e:
        print(f"Error with TinyLlama direct client: {e}")
        print(
            "This often means the model isn't served on the public Inference API for text-generation as expected."
        )

    print("\n--- Trying with a known working model (Mistral-7B-Instruct) ---")
    try:
        # Try with Mistral-7B-Instruct-v0.2 as a fallback
        client_mistral = InferenceClient(token=HF_TOKEN)
        response_mistral = client_mistral.text_generation(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            prompt="What is the capital of India?",
            max_new_tokens=50,
        )
        print(f"Mistral-7B-Instruct direct client response: {response_mistral}")

    except Exception as e:
        print(f"Error with Mistral-7B-Instruct direct client: {e}")
        print(
            "If this also fails, there might be a deeper issue with your token or network."
        )
