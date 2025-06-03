"""
Let say we have 5 docs now user will one question and the question is related
one of the document among the 5 document. We need to check from which document
the question belong.solve = generate the embedding of 5 docs as well as
generate the embeddings of the question so basically embedding is vectors so
here 5 vectors and 1 vector. In 300dm we have 5 vectors a new vector came we
need to find out that 1 vector is closed to which of the among 5 vector whose
similarity score is more is the answer.
"""

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

documents = [
    "Virat Kohli is a modern-day batting great known for his aggressive style and consistency across all formats."
    "Rohit Sharma is a stylish opener famed for his effortless strokeplay and record-breaking double centuries in ODIs."
    "Sachin Tendulkar is a Widely regarded as the greatest batsman of all time, with an unmatched legacy in international cricket."
    "MS Dhoni is alegendary captain and finisher, celebrated for his calm demeanor and leading India to major ICC trophies."
    "Jasprit Bumrah is a premier fast bowler known for his unorthodox action, lethal yorkers, and match-winning spells."
]

query = "tell me about virat kohli"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

# cosine similarity = 2D List

# print(cosine_similarity([query_embedding], doc_embeddings))

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(query)
print(documents[index])
print("simlarity score is:", score)
