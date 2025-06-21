"""
This script uses a local Falcon-RW 1B language model to generate and explain
jokes about a given topic.  It defines prompt templates and constructs chains
using LangChain's `RunnableSequence` and `RunnableParallel`. The model is
loaded via HuggingFace and wrapped in a text-generation pipeline. The
`joke_gen_chain` generates a joke, and then both the joke and its explanation
are produced in parallel. Finally, the chain is invoked with the topic
"cricket" and prints the resulting outputs.

         ┌──────────────────────────────┐
         │      Input: {"topic"}        │
         └────────────┬────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │   RunnableSequence          │
        │   (joke_gen_chain)          │
        │   [prompt1 → llm → parser]  │
        └────────────┬────────────────┘
                     │
                     ▼
       ┌───────────────────────────────┐
       │  RunnableParallel             │
       │ ┌───────────────────────────┐ │
       │ │ joke: RunnablePassthrough │ │
       │ └───────────────────────────┘ │
       │ ┌───────────────────────────┐ │
       │ │ explanation:              │ │
       │ │ [prompt2 → llm → parser]  │ │
       │ └───────────────────────────┘ │
       └─────────────┬─────────────────┘
                     │
                     ▼
           ┌─────────────────────┐
           │  Output:            │
           │ { joke, explanation}│
           └─────────────────────┘

"""

from langchain.schema.runnable import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableSequence,
)
from langchain_community.llms import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load local CPU model
model_name = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Wrap in a generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
)

# Create LangChain-compatible wrapper
llm = HuggingFacePipeline(pipeline=pipe)

# Prompt templates
prompt1 = PromptTemplate.from_template("Write a joke about {topic}")
prompt2 = PromptTemplate.from_template("Explain the following joke - {text}")

# Output parser
parser = StrOutputParser()

# Chains
joke_gen_chain = RunnableSequence(prompt1, llm, parser)

parallel_chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "explanation": RunnableSequence(prompt2, llm, parser),
    }
)

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

# Run the chain
output = final_chain.invoke({"topic": "cricket"})
print(output)
