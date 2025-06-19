"""
This script uses a locally available Falcon model with LangChain to
generate structured facts about a given topic. It defines a simple
schema for three factual outputs and formats the prompt accordingly.
The HuggingFace pipeline serves as the backend language model.
Finally, it prints the parsed response containing structured facts.
"""

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load locally available free model
model_id = "tiiuae/falcon-rw-1b"  # Small model, runs on CPU

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=pipe)

schema = [
    ResponseSchema(name="fact_1", description="Fact 1 about the topic"),
    ResponseSchema(name="fact_2", description="Fact 2 about the topic"),
    ResponseSchema(name="fact_3", description="Fact 3 about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give 3 fact about {topic}.\n{format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

chain = template | llm | parser

result = chain.invoke({"topic": "black hole"})
print(result)
