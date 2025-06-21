"""
This script scrapes a webpage, chunks the text, and answers a question using a 
CPU-based Hugging Face model. It avoids OpenAI and runs entirely on local 
machine using LangChain + Transformers.
"""

import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Set user-agent to avoid request blocking
os.environ["USER_AGENT"] = "Mozilla/5.0 (X11; Linux x86_64)"

# Load local CPU model
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, 
                device=-1)

llm = HuggingFacePipeline(pipeline=pipe)

# Prompt template
prompt = PromptTemplate(
    template="Answer the following question:\n{question}\n\nBased on this text:\n{text}",
    input_variables=["question", "text"]
)

parser = StrOutputParser()

# Load and chunk webpage
url = 'https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421'
loader = WebBaseLoader(url)
docs = loader.load()

# Split content into safe-size chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# Run chain on first chunk (or loop through all)
chain = prompt | llm | parser

# Just use the first chunk for now
response = chain.invoke({
    "question": "What is the product that we are talking about?",
    "text": split_docs[0].page_content
})

print("\n Answer:", response)
