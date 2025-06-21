"""
This script performs document question answering using fully open-source
components.It loads a local text file, splits the content into chunks, and
generates embeddings using a CPU-friendly Hugging Face model.These embeddings
are stored in a FAISS vector store to enable semantic search and retrieval.
A lightweight open-source LLM then generates an answer to a user query based
on the most relevant retrieved content.All components are CPU-compatible and
do not rely on proprietary services like OpenAI.
"""

import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load the document
loader = TextLoader("docs.txt")
documents = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Use CPU-compatible embeddings model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector store
vectorstore = FAISS.from_documents(docs, embedding_model)

# Create retriever
retriever = vectorstore.as_retriever()

# User query
query = "What are the key takeaways from the document?"
retrieved_docs = retriever.invoke(query)  # replaces .get_relevant_documents()

# Combine retrieved content
retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

# Load LLM (CPU mode)
model_id = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True,
    device=torch.device("cpu"),
)

# Initialize LangChain pipeline wrapper
llm = HuggingFacePipeline(pipeline=pipe)

# Create prompt
prompt = f"Based on the following text, answer the question:\n\nQuestion: {query}\n\nText:\n{retrieved_text}"

# Get LLM response
answer = llm.invoke(prompt)  # replaces .predict()
print("Answer:", answer.strip())
