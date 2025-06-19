"""
This script uses a Hugging Face model via LangChain to generate structured
data about a fictional person from a given place. It defines a Pydantic
schema to enforce output structure and validation. The `google/gemma-2-2b-it`
model is invoked using a prompt template. The final structured result
includes the person's name, age (over 18), and city.
"""

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel, Field

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(repo_id="google/gemma-2-2b-it", task="text-generation")

model = ChatHuggingFace(llm=llm)


class Person(BaseModel):

    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="Name of the city the person belongs to")


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name, age and city of a fictional {place} person \n {format_instruction}",
    input_variables=["place"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

chain = template | model | parser

final_result = chain.invoke({"place": "sri lankan"})

print(final_result)
