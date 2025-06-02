"""
For API key you need to visit ai.google.dev, It is a close source model 
so it is paid. model is in company server we can communicate only through api 
only so we don't have control to make changes.
"""


from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

result = model.invoke("What is the capital of India")

print(result.content)