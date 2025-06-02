"""
For API key you need to visit platform.openai.com, It is a close source model 
so it is paid. model is in company server we can communicate only through api 
only so we don't have control to make changes.
"""

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=1.5, max_completion_tokens=10)

result = model.invoke("Write a 5 line poem on cricket")

print(result.content)