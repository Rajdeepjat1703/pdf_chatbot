import streamlit as st
import requests

st.title("📄 PDF Chatbot (RAG + Gemini)")

API_URL = "http://localhost:8000"

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    files = {"file": uploaded_file.getvalue()}
    res = requests.post(f"{API_URL}/upload", files=files)
    st.success("PDF Uploaded!")

question = st.text_input("Ask a question from PDF")

if st.button("Ask"):
    res = requests.post(f"{API_URL}/chat", params={"question": question})
    st.write(res.json()["answer"])