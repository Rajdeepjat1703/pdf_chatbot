import streamlit as st
import requests

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 PDF Chatbot (RAG + Groq)")

API_URL = "http://localhost:8000"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

# Sidebar for PDF upload
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file and not st.session_state.pdf_uploaded:
        with st.spinner("Processing PDF..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            res = requests.post(f"{API_URL}/upload", files=files)
            if res.status_code == 200:
                st.session_state.pdf_uploaded = True
                st.session_state.messages = []  # reset chat on new PDF
                st.success("PDF uploaded successfully!")
            else:
                st.error("Upload failed.")

    if st.session_state.pdf_uploaded:
        st.success("✅ PDF Ready")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        requests.post(f"{API_URL}/clear")
        st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your PDF..."):
    if not st.session_state.pdf_uploaded:
        st.warning("Please upload a PDF first!")
    else:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                res = requests.post(f"{API_URL}/chat", json={
                    "question": prompt,
                    "chat_history": st.session_state.messages[:-1]  # exclude current question
                })
                answer = res.json()["answer"]
                st.write(answer)

        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": answer})