from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import streamlit as st

# Load the BlenderBot tokenizer and model
model_name = "facebook/blenderbot-1B-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# Streamlit UI
st.title("🧠 Open Source Chatbot - BlenderBot")
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
