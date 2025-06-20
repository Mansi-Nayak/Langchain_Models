"""
Blog Title Generator Using Open-Source CPU-Compatible LLM
This script takes a user-provided topic and generates a catchy blog title
using a local, open-source language model (e.g., Falcon-RW-1B) from Hugging
Face. It leverages LangChain’s LLMChain with a prompt template and runs
entirely on CPU, requiring no external APIs. Useful for quick content ideation
in offline or privacy-conscious environments.
"""

import torch
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load a small CPU-compatible LLM from Hugging Face
model_id = "tiiuae/falcon-rw-1b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

text_gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=50,
    temperature=0.7,
    do_sample=True,
    device=torch.device("cpu"),
)

# Wrap the Hugging Face pipeline in LangChain's interface
llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

# Define the prompt template
prompt = PromptTemplate(
    input_variables=["topic"], template="Suggest a catchy blog title about {topic}."
)

# Create the LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain
topic = input("Enter a topic: ")
output = chain.run(topic)

print("Generated Blog Title:", output.strip())
