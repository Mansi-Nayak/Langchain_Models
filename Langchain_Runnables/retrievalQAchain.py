"""
Document Question Answering using Local CPU-compatible LLM and Embeddings.
This script loads a text document, splits it into chunks, generates semantic
embeddings, stores them in a FAISS vector store, and retrieves the most
relevant content to answer a user query using a local Hugging Face language
model. All components are open-source and run on CPU.
"""

import torch
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load document
loader = TextLoader("docs.txt")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Create embeddings (CPU-friendly)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Store embeddings in FAISS
vectorstore = FAISS.from_documents(docs, embedding_model)

# Create retriever
retriever = vectorstore.as_retriever()

# Load local LLM (small and CPU-compatible)
model_id = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True,
    device=torch.device("cpu"),
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Ask question
query = "What are the key takeaways from the document?"
answer = qa_chain.run(query)

print("Answer:", answer.strip())
