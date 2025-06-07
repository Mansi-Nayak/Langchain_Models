"""
1. The code builds a Streamlit-based research tool where users can select a
    research paper, explanation style, and explanation length from dropdowns.
2. It uses load_prompt() to load a prompt template (likely stored in
    template.json) which defines how the explanation should be structured.
3. When the "Summarize" button is clicked, the selected inputs are passed
    to the prompt template.
4. Instead of manually formatting the prompt, the template and OpenAI model
    are combined into a chain (template | model) for cleaner execution.
5. The generated explanation is then displayed on the Streamlit page
    using st.write().
"""

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt
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

# the template file generate through the prompt_generator.py and then we use that template file here

template = load_prompt("template.json")

# fill the placeholders

# Reusable

# prompt = template.invoke(
#     {
#         "paper_input": paper_input,
#         "style_input": style_input,
#         "length_input": length_input,
#     }
# )

# if st.button("Summarize"):
#     result = model.invoke(prompt)
#     st.write(result.content)

# Chain              *when you need to use chain comment out the Reusable*

if st.button("Summarize"):
    chain = template | model
    result = chain.invoke(
        {
            "paper_input": paper_input,
            "style_input": style_input,
            "length_input": length_input,
        }
    )
    st.write(result.content)
