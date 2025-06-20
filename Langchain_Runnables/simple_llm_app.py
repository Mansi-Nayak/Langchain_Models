"""
This script generates a catchy blog title based on a user-provided topic using
an open-source language model.It loads the 'tiiuae/falcon-rw-1b' model from
Hugging Face and runs entirely on CPU, making it hardware-friendly.
LangChain's HuggingFacePipeline is used to integrate the model into a
prompt-based pipeline.The user inputs a topic, and the model outputs a
creative blog title suggestion.This setup avoids using proprietary APIs like
OpenAI and is fully open source.
"""

import torch
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model and tokenizer from Hugging Face
model_id = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Use CPU
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=50,
    temperature=0.7,
    do_sample=True,
    device=torch.device("cpu"),
)

# Initialize the LangChain HuggingFacePipeline LLM
llm = HuggingFacePipeline(pipeline=pipe)

# Create a prompt template
prompt = PromptTemplate(
    input_variables=["topic"], template="Suggest a catchy blog title about {topic}."
)

# Get user input
topic = input("Enter a topic: ")

# Format the prompt using PromptTemplate
formatted_prompt = prompt.format(topic=topic)

# Get prediction
blog_title = llm.predict(formatted_prompt)

# Output the result
print("Generated Blog Title:", blog_title.strip())
