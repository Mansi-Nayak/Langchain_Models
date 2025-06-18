"""
This script uses LangChain with a Hugging Face open-source model to generate a
report and its summary.It takes a topic as input, generates a detailed report
using a small language model (Falcon-RW-1B),then summarizes that report into
five concise lines.The script demonstrates how to compose multiple prompt
chains using LangChain's pipeline style.It is designed for educational and
inference purposes on lightweight models.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# Use a smaller model: Falcon-RW-1B
model = HuggingFaceEndpoint(
    repo_id="tiiuae/falcon-rw-1b",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.7,
    timeout=120,
)

# Prompt 1: Generate a detailed report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}.", input_variables=["topic"]
)

# Prompt 2: Generate 5-line summary from the report
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text:\n{text}",
    input_variables=["text"],
)

# Output parser
parser = StrOutputParser()

# Define chain 1: topic → report
report_chain = template1 | model | parser | template2 | model | parser

# Run chain
result = report_chain.invoke({"topic": "black hole"})

# Output final result
print(result)
