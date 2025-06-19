"""
This script uses the Falcon-RW-1B model from Hugging Face via LangChain to
extract a structured summary and sentiment from a product review. It
defines a TypedDict schema for the expected JSON output. The model is loaded
in a CPU-friendly way and wrapped using HuggingFacePipeline for text
generation. A system prompt guides the model to return the output in a
structured format, which is then printed.
"""

from typing import TypedDict

import torch
from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Define structured output schema


class Review(TypedDict):
    summary: str
    sentiment: str


# Load small Falcon model from Hugging Face (CPU-friendly)


model_name = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(torch.device("cpu"))

# Create generation pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
llm = HuggingFacePipeline(pipeline=pipe)

# Input to be analyzed
review_text = """The hardware is great, but the software feels bloated. \
There are too many pre-installed apps that I can't remove. \
Also, the UI looks outdated compared to other brands. Hoping for a software 
update to fix this."""

# System prompt that instructs the model to return structured output
system_prompt = (
    "You are an assistant that extracts structured information from reviews.\n"
    "Return a JSON object with the following keys:\n"
    "- summary: A brief summary of the review.\n"
    "- sentiment: One of 'positive', 'neutral', or 'negative'."
)

# Send messages to model
messages = [SystemMessage(content=system_prompt), HumanMessage(content=review_text)]

# Invoke model
response = llm.invoke(messages)
print(response)
