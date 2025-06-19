"""
This script uses Hugging Face's Zephyr-7B model via LangChain to generate
content.It first creates a 10-sentence explanation about a given topic using a
text-generation model.Then, it summarizes the generated content into a single
paragraph of exactly 5 sentences.The process involves two prompt templates and
interaction with the Hugging Face endpoint.The final output is a concise
summary suitable for quick understanding or report usage.
"""

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

# Setup model with timeout
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", task="text-generation", timeout=60
)

model = ChatHuggingFace(llm=llm)

# Shorter report to avoid timeout
template1 = PromptTemplate(
    template="Write a brief explanation about {topic} in 10 sentences.",
    input_variables=["topic"],
)

# Request a single paragraph 5-line summary
template2 = PromptTemplate(
    template=(
        "Summarize the following text into exactly 5 full sentences in a single paragraph. "
        "Avoid bullet points or numbering.\n\n"
        "Text:\n{text}\n\n"
        "Summary:"
    ),
    input_variables=["text"],
)

# Step 1: Generate shorter base text
prompt1 = template1.invoke({"topic": "black hole"}).to_string()
detailed_result = model.invoke([HumanMessage(content=prompt1)])

# Step 2: Condense into 5-line paragraph
prompt2 = template2.invoke({"text": detailed_result.content}).to_string()
summary_result = model.invoke([HumanMessage(content=prompt2)])

# Output the final result
print("\n Final 5-Sentence Summary (Single Paragraph):\n")
print(summary_result.content.strip())
