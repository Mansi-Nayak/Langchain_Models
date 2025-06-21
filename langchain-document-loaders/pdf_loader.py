"""
This script loads a PDF file using LangChain's PyPDFLoader to extract its 
content. It reads all the pages from the PDF and stores them as document 
objects. Each document includes both the text content and associated metadata.
The script then prints the total number of pages loaded and displays the
content of the first page along with metadata from the second page.
"""

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('dl-curriculum.pdf')

docs = loader.load()

print(len(docs))

print(docs[0].page_content)
print(docs[1].metadata)