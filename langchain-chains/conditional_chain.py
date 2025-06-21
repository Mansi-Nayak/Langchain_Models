"""
This script classifies sentiment from user feedback and generates a response
based on that sentiment.It uses google/flan-t5-base, which runs on CPU. If the
model fails to follow JSON output, we handle it manually.

                 RunnableBranch
                /       |       \
               /        |        \
              /         |         \
Condition:   x.sentiment == "positive"   -->   PromptTemplate (positive)
                                              |
                                              v
                                       HuggingFacePipeline
                                              |
                                              v
                                       StrOutputParser

Condition:   x.sentiment == "negative"   -->   PromptTemplate (negative)
                                              |
                                              v
                                       HuggingFacePipeline
                                              |
                                              v
                                       StrOutputParser

Default: (fallback) --> RunnableLambda (returns static fallback text)

"""

import json
from typing import Literal

from langchain.schema.runnable import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from pydantic import BaseModel, Field
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Load FLAN-T5 model (instruction-following)
model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
pipe = pipeline(
    "text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100
)
llm = HuggingFacePipeline(pipeline=pipe)

# Output string parser
parser = StrOutputParser()

# Define expected output schema


class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Sentiment classification"
    )


# Safe parser to fall back if JSON parsing fails


def safe_parse(output_str: str):
    try:
        data = json.loads(output_str)
        return Feedback(**data)
    except Exception:
        stripped = output_str.strip().lower()
        if "positive" in stripped:
            return Feedback(sentiment="positive")
        elif "negative" in stripped:
            return Feedback(sentiment="negative")
        else:
            raise ValueError(f" Could not parse output: {output_str}")


# Prompt to classify feedback
prompt1 = PromptTemplate(
    template=(
        "Classify the sentiment of this feedback as either positive or negative.\n"
        'Return ONLY a valid JSON like: {{"sentiment": "positive"}} or {{"sentiment": "negative"}}\n\n'
        "Feedback: {feedback}"
    ),
    input_variables=["feedback"],
)

# Prompts for responses
prompt_positive = PromptTemplate(
    template="Write a reply to this positive feedback:\n{feedback}",
    input_variables=["feedback"],
)

prompt_negative = PromptTemplate(
    template="Write a reply to this negative feedback:\n{feedback}",
    input_variables=["feedback"],
)

# Conditional branch for different responses
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt_positive | llm | parser),
    (lambda x: x.sentiment == "negative", prompt_negative | llm | parser),
    RunnableLambda(lambda x: "Could not classify sentiment."),
)

# ---- MAIN TEST ----

# Example input
feedback_text = "This is a beautiful phone"

# Step 1: Get raw model output for classification
raw_output = (prompt1 | llm).invoke({"feedback": feedback_text})
print("🔍 Raw LLM Output:", raw_output)

# Step 2: Parse classification
parsed = safe_parse(raw_output)
print(" Parsed Sentiment:", parsed)

# Step 3: Generate final reply
result = branch_chain.invoke(parsed)
print("\n Final Response:")
print(result)

# Optional: Chain graph
print("\n Chain Graph:")
branch_chain.get_graph().print_ascii()
