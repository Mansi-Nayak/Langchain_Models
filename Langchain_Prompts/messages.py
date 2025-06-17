"""
This code sets up a lightweight Falcon language model (`falcon-rw-1b`) using
Hugging Face Transformers and runs it on the CPU for efficiency. It uses a
text generation pipeline and wraps it with LangChain's `HuggingFacePipeline`
to allow message-based interactions. A simple conversation is initialized with
a system message (setting the assistant's behavior) and a human message asking
about LangChain. The model processes the input and generates a response, which
is then added to the conversation history. Finally, all messages (system,
human, and AI) are printed out in order.
"""

import torch
from langchain_community.llms import HuggingFacePipeline
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load a smaller, CPU-friendly Falcon model
model_name = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(torch.device("cpu"))  # Force model to run on CPU

# Create a text generation pipeline (CPU)
text_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

# Wrap it with LangChain interface
llm = HuggingFacePipeline(pipeline=text_pipeline)

# Conversation messages
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about LangChain"),
]

# Invoke the model
response = llm.invoke(messages)

# Append the AI's response
messages.append(AIMessage(content=response))

# Print the full conversation
for msg in messages:
    print(f"{msg.type.capitalize()}: {msg.content}")
