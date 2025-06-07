"""
1. It uses Streamlit to build a web interface and loads environment variables
   using load_dotenv().
2. It initializes an OpenAI language model (gpt-3.5-turbo) using LangChain with
   a creativity setting (temperature=0.7).
3. A text input box collects the user’s prompt, and a button labeled
   "Summarize" triggers the model.
4. When clicked, the model generates a response based on the prompt using
   model.invoke().
5. The generated summary is then displayed using st.write(), or a warning is
   shown if input is empty.
"""

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

st.header("Research Tool - Summarize Anything")

# static
user_input = st.text_input("Enter your prompt")

if st.button("Summarize"):
    if user_input:
        result = model.invoke(user_input)
        st.write(result.content)
    else:
        st.warning("Please enter a prompt.")
