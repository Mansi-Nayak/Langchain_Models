"""
This script uses LangChain with HuggingFace's `Mistral-7B-Instruct` model
to generate structured JSON output. It prompts the model to return 5 facts
about a specified topic using a defined format. The JsonOutputParser ensures
the output conforms to a valid JSON list of facts. The result is printed
after the full prompt → model → parse pipeline is executed.
"""

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()

# Initialize LLM directly (not chat model)
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
    timeout=60,
)

parser = JsonOutputParser()

# template = PromptTemplate(
#     template="Give me the name, age and city of a fictional person \n {format_instruction}",
#     input_variables=[],
#     partial_variables={"format_instruction": parser.get_format_instructions()},
# )

# chain = template | model | parser

# result = chain.invoke({})

# print(result)

# json don't support schema enforce

template = PromptTemplate(
    template="Give me 5 facts about {topic} \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

chain = template | llm | parser

result = chain.invoke({"topic": "black hole"})

print(result)
