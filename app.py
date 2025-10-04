# app.py
import streamlit as st
from src.chatbot import LegalChatbot
from src.utils import load_env

st.set_page_config(page_title="⚖️ Legal Chatbot", layout="centered")

@st.cache_resource
def get_chatbot(index_path="./faiss_index",
                embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
                llm_model_name="google/flan-t5-base"):
    return LegalChatbot(index_path=index_path,
                        embedding_model_name=embedding_model_name,
                        llm_model_name=llm_model_name)

def main():
    st.title("⚖️ Legal Chatbot")

    load_env()

    st.sidebar.markdown("### ⚙️ Settings")
    mode = st.sidebar.selectbox("Choose mode", ["Normal (Plain answers)", "Advanced (Show sources)"])
    top_k = st.sidebar.slider("Retrieval: top_k", min_value=1, max_value=15, value=5)
    min_score = st.sidebar.slider("Similarity threshold", min_value=0.10, max_value=0.90, value=0.25, step=0.01)

    chatbot = get_chatbot()

    if "history" not in st.session_state:
        st.session_state.history = []

    user_input = st.chat_input("Ask a legal question (e.g., 'What are Fundamental Rights?')")

    if user_input and user_input.strip():
        with st.spinner("🤔 Thinking..."):
            resp = chatbot.ask(
                user_input,
                top_k=top_k,
                show_sources=(mode == "Advanced (Show sources)"),
                min_score=min_score
            )
            st.session_state.history.append((user_input, resp))

    for q, a in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)

if __name__ == "__main__":
    main()
