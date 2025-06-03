import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

st.header("🧠 Research Tool - Summarize Anything")

# static
user_input = st.text_input("Enter your prompt")

if st.button("Summarize"):
    if user_input:
        result = model.invoke(user_input)
        st.write(result.content)
    else:
        st.warning("Please enter a prompt.")
