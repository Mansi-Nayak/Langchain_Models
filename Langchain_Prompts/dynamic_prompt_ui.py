"""
This code creates a simple Streamlit web app for a research paper 
summarization tool. It first loads environment variables and initializes an 
OpenAI language model using LangChain. The interface includes three dropdown 
menus where the user can select a research paper title, the explanation style 
(e.g., beginner or technical), and the desired explanation length. When the 
"Summarize" button is clicked, it currently just prints "Hello" in the 
terminal, indicating where the summarization logic would be added later. 
The UI is ready, but the model response generation is yet to be implemented.
"""

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI()

st.header("Research Tool")

# Dynamic

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis",
    ],
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"],
)

length_input = st.selectbox(
    "Select Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (detailed explanation)",
    ],
)


if st.button("Summarize"):
    print("Hello")
