"""
For API key you need to visit console.anthropic.com, It is a close source model 
so it is paid. model is in company server we can communicate only through api 
only so we don't have control to make changes.
"""

from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model='claude-3-5-sonnet-20241022')

result = model.invoke("What is the capital of India")


print(result.content)