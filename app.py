# streamlit_agent_ui.py

import streamlit as st
from testAgent import agent_executor, initial_state, PDF_CHUNKS

st.set_page_config(page_title="NCLEX AI Nurse", layout="centered")

st.title("NCLEX AI Nursing Assistant")
st.markdown("Ask me anything from the nursing Q&A PDF!")

# Show error if no content
if not PDF_CHUNKS:
    st.error("No PDF content loaded. Please check the parsing in `testAgent.py`.")
    st.stop()

# Initialize session state
if "state" not in st.session_state:
    st.session_state.state = initial_state.copy()

if "messages" not in st.session_state:
    st.session_state.messages = []  # List of (user, bot) message tuples

# Display chat messages
for i, (user_msg, bot_msg) in enumerate(st.session_state.messages):
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(bot_msg)

# Accept new user input
if prompt := st.chat_input("Ask a nursing-related question..."):
    # Display user message in chat
    st.chat_message("user").markdown(prompt)

    # Prepare agent input
    st.session_state.state["user_input"] = prompt
    st.session_state.state["retry"] = False
    st.session_state.state["retry_count"] = 0

    with st.spinner("Thinking like a nurse... "):
        result_state = agent_executor.invoke(st.session_state.state)
        st.session_state.state["message_history"] = result_state.get("message_history", [])
        response = result_state["response"]
        chunks = result_state["retrieved_chunks"]

    # Display assistant response
    st.chat_message("assistant").markdown(response)

    # Store messages for ongoing chat
    st.session_state.messages.append((prompt, response))

    # Optional: debug retrieved chunks
    with st.expander("Retrieved Chunks (debug)"):
        for i, chunk in enumerate(chunks, 1):
            st.markdown(f"**Chunk {i}:**\n\n{chunk}")
