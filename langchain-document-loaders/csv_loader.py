"""
This script loads data from a CSV file using LangChain's CSVLoader. 
Each row in the CSV is treated as a separate document with its content and 
metadata. The script loads all rows into memory as document objects. 
It prints the total number of documents loaded and displays the second 
document's content. Useful for processing tabular data in a document-oriented 
pipeline.
"""

from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='Social_Network_Ads.csv')

docs = loader.load()

print(len(docs))
print(docs[1])