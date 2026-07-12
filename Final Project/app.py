"""
app.py — Streamlit UI for IITB Insti-Assist (Academic Assistant).

Run with:  streamlit run app.py

Requires GROQ_API_KEY to be set as an environment variable, or entered
in the sidebar at runtime. Get a free key at https://console.groq.com/keys
"""

import os
from pathlib import Path

import streamlit as st

from rag import InstiAssistRAG

st.set_page_config(page_title="IITB Insti-Assist — Academic Assistant", page_icon="🎓", layout="centered")

st.title("🎓 IITB Insti-Assist")
st.caption("A RAG-powered academic assistant for IIT Bombay — ask about registration, grading, CPI/SPI, exams, medals, and academic malpractice rules.")

with st.sidebar:
    st.header("Setup")
    api_key_input = st.text_input("Groq API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""))
    st.caption("Free key: https://console.groq.com/keys")
    st.markdown("---")
    st.markdown("**Knowledge base (5 official IITB documents):**")
    st.markdown(
        "- UG Rules & Regulations (June 2025)\n"
        "- Academic Calendar 2025-26\n"
        "- Rules for Medals & Academic Prizes\n"
        "- Academic Malpractice Punishments\n"
        "- M.Sc./PG Rules & Regulations"
    )
    st.markdown("---")
    st.caption("This assistant only answers from the documents above. It will say 'I don't know' for anything outside their scope (e.g. hostel rules, club activities).")

if not Path("data/index/faiss.index").exists():
    st.error("No index found. Run `python ingest.py` first to build the knowledge base.")
    st.stop()

if "rag" not in st.session_state:
    if not api_key_input:
        st.info("Enter your Groq API key in the sidebar to get started.")
        st.stop()
    with st.spinner("Loading knowledge base..."):
        st.session_state.rag = InstiAssistRAG(groq_api_key=api_key_input)

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            with st.expander("📄 Sources used"):
                for s in turn["sources"]:
                    st.markdown(f"- **{s['label']}** (relevance: {s['score']})")

query = st.chat_input("Ask about IITB academics, e.g. 'What is the minimum CPI to avoid ARP?'")

if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base and generating answer..."):
            result = st.session_state.rag.answer(query)
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("📄 Sources used"):
                for s in result["sources"]:
                    st.markdown(f"- **{s['label']}** (relevance: {s['score']})")

    st.session_state.history.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
