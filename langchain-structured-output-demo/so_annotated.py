"""
This script uses the Falcon-RW-1B model from Hugging Face via LangChain to
extract a structured summary and sentiment from a product review. It
defines a TypedDict schema for the expected JSON output. The model is loaded
in a CPU-friendly way and wrapped using HuggingFacePipeline for text
generation. A system prompt guides the model to return the output in a
structured format, which is then printed.
"""

from typing import Annotated, Literal, Optional, TypedDict

import torch
from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Define structured output schema


class Review(TypedDict):
    key_themes: Annotated[
        list[str], "Write down all the key themes discussed in " "the review in a list"
    ]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[
        Literal["pos", "neg"],
        "Return sentiment of the review either negative, " "positive or neutral",
    ]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]


# Load small Falcon model from Hugging Face (CPU-friendly)


model_name = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(torch.device("cpu"))

# Create generation pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
llm = HuggingFacePipeline(pipeline=pipe)

# Input to be analyzed
review = """
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don’t use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Cons:
Bulky and heavy—not great for one-handed use
Bloatware still exists in One UI
Expensive compared to competitors

"""

# System prompt that instructs the model to return structured output
system_prompt = (
    "You are an assistant that extracts structured information from reviews.\n"
    "Return a JSON object with the following keys:\n"
    "- summary: A brief summary of the review.\n"
    "- sentiment: One of 'positive', 'neutral', or 'negative'."
)

# Send messages to model
messages = [SystemMessage(content=system_prompt), HumanMessage(content=review)]

# Invoke model
response = llm.invoke(messages)
print(response)
