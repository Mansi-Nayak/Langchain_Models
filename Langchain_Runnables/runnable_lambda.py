"""
This script builds a LangChain pipeline using the local Falcon-RW 1B model to
generate a joke about a given topic. It wraps the model using HuggingFace's
`pipeline` for CPU inference and formats the response using prompt templates.
After generating a joke, it simultaneously passes the joke and calculates its
word count using a parallel chain.The final output includes both the joke and
its word count.The result is printed after invoking the chain with the topic
"AI".

         ┌────────────────────────────┐
         │   Input: {"topic": "AI"}   │
         └────────────┬──────────────┘
                      │
                      ▼
       ┌───────────────────────────────┐
       │ RunnableSequence              │
       │ [prompt → llm → parser]       │
       └────────────┬──────────────────┘
                    │
                    ▼
     ┌────────────────────────────────────┐
     │           RunnableParallel         │
     │                                    │
     │  ┌──────────────────────────────┐  │
     │  │  joke: RunnablePassthrough   │  │
     │  └──────────────────────────────┘  │
     │  ┌──────────────────────────────┐  │
     │  │ word_count: RunnableLambda   │  │
     │  │     (counts words in joke)   │  │
     │  └──────────────────────────────┘  │
     └──────────────────┬────────────────┘
                        │
                        ▼
           ┌────────────────────────────┐
           │  Output:                   │
           │  {joke, word_count}        │
           └────────────────────────────┘

"""

from langchain.schema.runnable import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSequence,
)
from langchain_community.llms import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load local Hugging Face model (CPU-compatible)
model_name = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
)

# LangChain-compatible wrapper
llm = HuggingFacePipeline(pipeline=pipe)

# Count words in generated joke


def word_count(text):
    return len(text.split())


# Prompt and output parser


prompt = PromptTemplate.from_template("Write a joke about {topic}")
parser = StrOutputParser()

# Define LangChain pipeline
joke_gen_chain = RunnableSequence(prompt, llm, parser)

parallel_chain = RunnableParallel(
    {"joke": RunnablePassthrough(), "word_count": RunnableLambda(word_count)}
)

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

# Invoke chain
result = final_chain.invoke({"topic": "AI"})

# Format output
final_result = "{} \nword count - {}".format(result["joke"], result["word_count"])

print(final_result)
