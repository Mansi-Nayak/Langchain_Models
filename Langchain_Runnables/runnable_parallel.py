"""
This script uses an open-source language model to generate a tweet and a
LinkedIn post about a given topic. It runs entirely on CPU using HuggingFace
models. Two separate prompt chains are executed in parallel using LangChain's
RunnableParallel. The Falcon-RW-1B model is used via HuggingFacePipeline for
local inference. Finally, it prints both the tweet and LinkedIn post.

+-----------------------+
|     User Input        |
|   {'topic': 'AI'}     |
+-----------+-----------+
            |
            v
.------------------------------.
|     RunnableParallel         |
|    ┌────────────┐            |
|    |   Tweet     |           |
|    |  Sub-Chain  |           |
|    └────────────┘           |
|    ┌───────────────┐        |
|    |   LinkedIn     |       |
|    |   Sub-Chain    |       |
|    └───────────────┘        |
'------------------------------'
            |
            v
         Parallel Execution
            |
  .----------------.     .-----------------.
  | PromptTemplate |     | PromptTemplate  |
  | (Tweet prompt) |     | (LinkedIn post) |
  '----------------'     '-----------------'
            |                     |
            v                     v
    .---------------.      .----------------.
    |    Falcon RW  |      |    Falcon RW   |
    | (LLM on CPU)  |      |  (LLM on CPU)  |
    '---------------'      '----------------'
            |                     |
            v                     v
.--------------------.   .---------------------.
|  Output Parser     |   |  Output Parser      |
|  (Tweet string)    |   |  (LinkedIn string)  |
'--------------------'   '---------------------'
            |                     |
            +----------+----------+
                       v
        +------------------------------+
        | Final Result Dictionary      |
        | {'tweet': <text>,            |
        |  'linkedin': <text>}         |
        +------------------------------+

Output:
Tweet:
  <Generated tweet about AI>

LinkedIn Post:
  <Generated LinkedIn post about AI>

"""

import torch
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel, RunnableSequence
from langchain_community.llms import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

load_dotenv()

# Load a small CPU-friendly model
model_id = "tiiuae/falcon-rw-1b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    temperature=0.7,
    device=torch.device("cpu"),
    pad_token_id=tokenizer.eos_token_id,
)

llm = HuggingFacePipeline(pipeline=pipe)

# Prompt Templates
prompt1 = PromptTemplate(
    template="Generate a short and witty tweet about {topic}.",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Write a professional LinkedIn post about {topic}.",
    input_variables=["topic"],
)

# Output parser
parser = StrOutputParser()

# Parallel Chain
parallel_chain = RunnableParallel(
    {
        "tweet": RunnableSequence(prompt1, llm, parser),
        "linkedin": RunnableSequence(prompt2, llm, parser),
    }
)

# Invoke the chain
result = parallel_chain.invoke({"topic": "AI"})

# Print outputs
print("Tweet:\n", result["tweet"])
print("\nLinkedIn Post:\n", result["linkedin"])
