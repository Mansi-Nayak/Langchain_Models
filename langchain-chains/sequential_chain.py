"""
This script uses a locally hosted Hugging Face model (falcon-rw-1b) to
generate content on CPU.It first generates a detailed report about a given
topic using LangChain and the local model.Then, it summarizes the report into
5 key points using a second prompt.The model operates fully offline and
requires no external API keys.The output and the LangChain pipeline graph are
printed at the end.
"""

from langchain_community.llms import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load tokenizer and model
model_id = "tiiuae/falcon-rw-1b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Setup HuggingFace pipeline for text generation
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)

# Connect it to LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# Prompt template
prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}", input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following text \n {text}",
    input_variables=["text"],
)

# Output parser
parser = StrOutputParser()

# Create the chain
chain = prompt1 | llm | parser | prompt2 | llm | parser

# Run the chain
result = chain.invoke({"topic": "unemployment in India"})

# Print result
print(result)

# Optional: Print the pipeline graph
chain.get_graph().print_ascii()
