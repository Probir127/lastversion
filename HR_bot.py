import streamlit as st
import random
import time
import backend


# Streamed response emulator
def response_generator(prompt):
    # Extract the most recent person name mentioned
    name = backend.extract_person_name(prompt)
    if name:
        st.session_state.last_person = name

    # Call backend with memory
    response = backend.get_response(
        user_input=prompt,
        chat_history=st.session_state.messages,
        last_person=st.session_state.get("last_person")
    )

    yield response

# for char in response:
st.set_page_config(
    page_title="HR Chatbot",
    page_icon="hr_icon.png"
)

#=== Page Title ===
st.title("HR chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_person" not in st.session_state:
    st.session_state.last_person = None
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})