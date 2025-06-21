"""
This script builds a LangChain pipeline using an open-source HuggingFace model
that runs on CPU. It first generates a joke based on a given topic using the
Falcon-RW-1B language model. Then, it passes the joke into another prompt
asking the model to explain it. The HuggingFace pipeline is used for local
text generation, avoiding the need for external APIs. The chain is executed
end-to-end and prints the final explanation output.

+----------------+
|   User Input   |
| {'topic': 'AI'}|
+-------+--------+
        |
        v
.------------------------.
|   Prompt Template 1    |  --> "Write a joke about AI"
'------------------------'
        |
        v
.------------------------.
|     LLM (Falcon RW)    |  --> Generates a joke
'------------------------'
        |
        v
.------------------------.
|   Output Parser (str)  |  --> Extracts plain string
'------------------------'
        |
        v
.------------------------.
|   Prompt Template 2    |  --> "Explain the following joke - <joke>"
'------------------------'
        |
        v
.------------------------.
|     LLM (Falcon RW)    |  --> Generates explanation
'------------------------'
        |
        v
.------------------------.
|   Output Parser (str)  |  --> Extracts explanation text
'------------------------'
        |
        v
+-----------------------+
| Final Explanation Out |
+-----------------------+

"""

import torch
from langchain.schema.runnable import RunnableSequence
from langchain_community.llms import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model and tokenizer (small model suitable for CPU)
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
)

llm = HuggingFacePipeline(pipeline=pipe)

# Prompt Templates
prompt1 = PromptTemplate(
    template="Write a joke about {topic}", input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Explain the following joke - {text}", input_variables=["text"]
)

# Output parser
parser = StrOutputParser()

# Runnable Chain
chain = RunnableSequence(prompt1, llm, parser, prompt2, llm, parser)

# Run the chain
print(chain.invoke({"topic": "AI"}))
