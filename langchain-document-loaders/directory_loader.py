"""
This script loads all PDF files from a specified directory using LangChain's 
DirectoryLoader. It applies PyPDFLoader to each matching file (based on the 
glob pattern) to extract content lazily. The documents are loaded one at a 
time to optimize memory usage for large collections. For each loaded document, 
the script prints its metadata (e.g., source file path and page number). 
Useful for inspecting a folder of PDFs without fully loading all content at 
once.
"""

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.lazy_load()

for document in docs:
    print(document.metadata)