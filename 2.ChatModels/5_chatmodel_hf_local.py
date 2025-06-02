# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline


# llm = HuggingFacePipeline.from_model_id(
#     model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation",
#     pipeline_kwargs=dict(
#         temperature=0.5,
#         max_new_tokens=100
#     )
# )
# model = ChatHuggingFace(llm=llm)

# result = model.invoke("What is the capital of India")

# print(result.content)

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import os
from dotenv import load_dotenv

load_dotenv()  # loads HUGGINGFACE_TOKEN from .env

llm = HuggingFaceEndpoint(
    repo_id="TheBloke/TinyLlama-1.1B-Chat-GGML",  # <-- your model repo here
    task="text-generation",
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN")  # if your repo is gated/private
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India?")
print(result.content)
