"""
For API key you need to visit platform.openai.com, It is a close source model
so it is paid. model is in company server we can communicate only through api
only so we don't have control to make changes.
"""

from dotenv import load_dotenv
from langchain_openai import OpenAI

load_dotenv()

llm = OpenAI(model="gpt-3.5-turbo-instruct")

result = llm.invoke("What is the capital of India")


print(result)
