"""
This script generates 5 interesting facts about a given topic using a local
language model.It uses the Hugging Face 'falcon-rw-1b' model to run fully on
CPU, requiring no internet API keys.LangChain's PromptTemplate and
OutputParser are used to construct and parse the generation chain.
The HuggingFacePipeline connects the model to LangChain seamlessly.
The result is printed, and the execution chain graph is optionally displayed.
"""

from langchain_community.llms import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load tokenizer and model (runs on CPU)
model_id = "tiiuae/falcon-rw-1b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Setup HuggingFace pipeline for text generation
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)

# Connect it to LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# Prompt template
prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}", input_variables=["topic"]
)

# Output parser
parser = StrOutputParser()

# Create the chain
chain = prompt | llm | parser

# Run the chain
result = chain.invoke({"topic": "cricket"})

# Print result
print(result)

# Optional: Print the pipeline graph
chain.get_graph().print_ascii()
