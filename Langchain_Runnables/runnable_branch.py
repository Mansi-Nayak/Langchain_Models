"""
This script builds a LangChain pipeline that generates a detailed report on a
given topic using the Falcon-RW 1B model. It conditionally summarizes the
generated report if its word count exceeds 300, using a branching logic.
Prompt templates are defined for both report generation and summarization, and
outputs are parsed to plain text. The HuggingFace model is run on CPU using a
text generation pipeline and wrapped with LangChain. The final result is
printed after invoking the chain with the topic "Russia vs Ukraine".

         ┌──────────────────────────────┐
         │ Input: {"topic": "Russia vs Ukraine"} │
         └────────────┬────────────────┘
                      ▼
     ┌────────────────────────────────────┐
     │     report_gen_chain               │
     │ [prompt1 → llm → parser]           │
     └────────────────┬──────────────────┘
                      │
                      ▼
           ┌──────────────────────────────┐
           │       branch_chain           │
           │ ┌──────────────────────────┐ │
           │ │ if len(text) > 300:      │ │
           │ │    [prompt2 → llm → parser]│
           │ └──────────────────────────┘ │
           │ ┌──────────────────────────┐ │
           │ │ else: passthrough        │ │
           │ └──────────────────────────┘ │
           └────────────┬────────────────┘
                        ▼
           ┌──────────────────────────────┐
           │        Output: Text          │
           └──────────────────────────────┘

"""

from langchain.schema.runnable import (
    RunnableBranch,
    RunnablePassthrough,
    RunnableSequence,
)
from langchain_community.llms import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load Hugging Face model (runs on CPU)
model_name = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    do_sample=True,
    temperature=0.7,
)

# LangChain-compatible wrapper
llm = HuggingFacePipeline(pipeline=pipe)

# Prompts
prompt1 = PromptTemplate.from_template("Write a detailed report on {topic}")
prompt2 = PromptTemplate.from_template("Summarize the following text \n {text}")

# Output parser
parser = StrOutputParser()

# Chain 1: Generate report
report_gen_chain = prompt1 | llm | parser

# Chain 2: Conditionally summarize if text is too long (> 300 words)
branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300, prompt2 | llm | parser), RunnablePassthrough()
)

# Combine both into final chain
final_chain = RunnableSequence(report_gen_chain, branch_chain)

# Invoke
output = final_chain.invoke({"topic": "Russia vs Ukraine"})
print(output)
