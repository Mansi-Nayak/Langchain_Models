"""
1. It uses Streamlit to create a chatbot interface and loads BlenderBot
   (a pre-trained open-source conversational AI model from Facebook).
2. The tokenizer converts user input into a format the model understands,
   and the model generates a response.
3. A text input box takes the user's message, and a Send button
   triggers the response generation.
4. Each user input and bot reply is saved in st.session_state.history
   to preserve the chat history.
5. The conversation history is shown line by line using Streamlit's st.write.
"""

import streamlit as st
from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer

# Load the BlenderBot tokenizer and model
model_name = "facebook/blenderbot-1B-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# Streamlit UI
st.title("Open Source Chatbot - BlenderBot")
st.write("Talk to an open-source conversational model!")

# Initialize chat history in session state
if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You:")

if st.button("Send"):
    if user_input:
        # Tokenize input
        inputs = tokenizer(user_input, return_tensors="pt")
        reply_ids = model.generate(**inputs)
        reply = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]

        # Store the conversation
        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Bot", reply))

# Display conversation history
for speaker, text in st.session_state.history:
    st.write(f"**{speaker}:** {text}")
