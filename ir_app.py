# streamlit_ui.py
import streamlit as st
from backend_kawasaki import load_vectorstore, create_qa_chain, handle_query
from langchain.schema import HumanMessage, AIMessage

# ---------------------- Page Config ----------------------
st.set_page_config(page_title="Kawasaki Robot Chatbot", layout="centered")

# ---------------------- Custom CSS -----------------------
CUSTOM_CSS = """
<style>
.chat-bubble-user {
    background-color: #DCF8C6;
    border-radius: 16px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 80%;
    text-align: right;
    float: right;
    clear: both;
}
.chat-bubble-ai {
    background-color: #F1F0F0;
    border-radius: 16px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 80%;
    text-align: left;
    float: left;
    clear: both;
}
.suggestion-chip {
    display: inline-block;
    background-color: #e8e8e8;
    padding: 6px 14px;
    border-radius: 20px;
    margin: 4px;
    cursor: pointer;
    font-size: 14px;
    border: 1px solid #ccc;
}
.suggestion-chip:hover {
    background-color: #dcdcdc;
}
.input-container {
    position: fixed;
    bottom: 10px;
    left: 50%;
    transform: translateX(-50%);
    width: 80%;
    background: white;
    padding: 10px 0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------- Main UI ----------------------
def run_streamlit_ui():
    st.title(" Kawasaki Robot Chat Assistant")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Load backend components once
    if "qa_chain" not in st.session_state:
        vectorstore, embeddings = load_vectorstore()
        st.session_state.qa_chain, st.session_state.memory = create_qa_chain(vectorstore)

    # Render chat messages
    st.markdown("### Chat")
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            st.markdown(f"<div class='chat-bubble-user'>{msg.content}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble-ai'>{msg.content}</div>", unsafe_allow_html=True)

    # Suggestion chips
    if "last_suggestions" in st.session_state and st.session_state.last_suggestions:
        st.markdown("###  Suggested Queries")
        for sug in st.session_state.last_suggestions:
            if st.button(sug, key=sug):
                # If user clicks suggestion chip, auto-fill and process
                user_query = sug
                response, suggestions, relevance, sources = handle_query(
                    user_query, st.session_state.qa_chain
                )
                st.session_state.chat_history.append(HumanMessage(content=user_query))
                st.session_state.chat_history.append(AIMessage(content=response))
                st.session_state.last_suggestions = suggestions
                st.rerun()

    # Fixed bottom input
    with st.container():
        st.markdown("<div class='input-container'>", unsafe_allow_html=True)
        user_query = st.text_input("Ask a question:", key="user_input", label_visibility="collapsed")
        submit = st.button("Send", key="send_btn")
        st.markdown("</div>", unsafe_allow_html=True)

    # If user sends a query
    if submit and user_query:
        response, suggestions, relevance, sources = handle_query(user_query, st.session_state.qa_chain)

        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

        # Store suggestions for future chips
        st.session_state.last_suggestions = suggestions

        st.rerun()


if __name__ == "__main__":
    run_streamlit_ui()
