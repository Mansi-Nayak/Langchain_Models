"""
This script loads a long poem from a text file and summarizes it using a local
Hugging Face model running on CPU. The poem is split into smaller chunks to fit 
the model's input size limit. Each chunk is summarized separately, and all 
summaries are then combined into a final summary. It uses LangChain components 
for prompt handling, text splitting, and model interaction.
"""

from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load Hugging Face model that runs on CPU
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, 
                device=-1)
llm = HuggingFacePipeline(pipeline=pipe)

# Prompt template
prompt = PromptTemplate(
    template="Write a summary for the following poem:\n\n{poem}",
    input_variables=["poem"]
)

parser = StrOutputParser()

# Load and split document into smaller chunks
loader = TextLoader("cricket.txt", encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      
    chunk_overlap=50    
)

split_docs = splitter.split_documents(docs)

# Build the chain
chain = prompt | llm | parser

# Summarize each chunk
summaries = []
for i, chunk in enumerate(split_docs):
    print(f"\n Processing chunk {i+1}/{len(split_docs)}...\n")
    summary = chain.invoke({"poem": chunk.page_content})
    print(summary)
    summaries.append(summary)

# Combine all chunk summaries
final_summary = "\n".join(summaries)

print("\n Final Combined Summary:\n")
print(final_summary)