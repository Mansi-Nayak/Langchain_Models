"""
This code sets up a LangChain chat prompt for a customer support agent.
It defines a chat template with a system role, a placeholder for previous
messages (chat_history), and a user query. It then reads past messages from a
file (chat_history.txt) into a list to preserve conversation context. Using
this history and a new user question ("where is my refund"), it generates a
prompt with chat_template.invoke() and prints both the chat history and the
final prompt.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Chat Template
chat_template = ChatPromptTemplate(
    [
        ("system", "you are a helpful customer support agent"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}"),
    ]
)

chat_history = []
# load chat history
with open("Langchain_Prompts/chat_history.txt") as f:
    chat_history.extend(f.readlines())

print(chat_history)

# create prompt
prompt = chat_template.invoke(
    {"chat_history": chat_history, "query": "where is my refund"}
)

print(prompt)
