"""
This code creates a Streamlit chatbot interface using Facebook’s open-source
BlenderBot model. It loads the model and tokenizer, and initializes a
conversation history in st.session_state starting with a system message
defining the bot’s role. When the user enters a message and clicks "Send", the
input is tokenized and passed to the BlenderBot model to generate a response.
Both the user’s message and the bot’s reply are stored as HumanMessage and
AIMessage objects respectively. The app then loops through the conversation
history and neatly displays each message in the chat format.
"""

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from transformers import (BlenderbotForConditionalGeneration,
                          BlenderbotTokenizer)

# Load the BlenderBot tokenizer and model
model_name = "facebook/blenderbot-1B-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# Streamlit UI
st.title("Open Source Chatbot - BlenderBot")
st.write("Talk to an open-source conversational model!")

# Initialize chat history in session state
if "history" not in st.session_state:
    st.session_state.history = [SystemMessage
                                (content="You are a helpful AI assistant")]

user_input = st.text_input("You:")

if st.button("Send"):
    if user_input:
        # Append user's message as a HumanMessage
        st.session_state.history.append(HumanMessage(content=user_input))

        # Tokenize input (Note: BlenderBot doesn't directly use Langchain
        # message objects for input)
        # We'll extract the content from the history to pass to the tokenizer
        # For simplicity, let's just use the latest user input for the
        # BlenderBot model
        inputs = tokenizer(user_input, return_tensors="pt")
        reply_ids = model.generate(**inputs)
        reply_content = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]

        # Append bot's reply as an AIMessage
        st.session_state.history.append(AIMessage(content=reply_content))

# Display conversation history
# Iterate through the history and display content based on message type
for message in st.session_state.history:
    if isinstance(message, SystemMessage):
        st.write(f"**System:** {message.content}")
    elif isinstance(message, HumanMessage):
        st.write(f"**You:** {message.content}")
    elif isinstance(message, AIMessage):
        st.write(f"**AI:** {message.content}")
