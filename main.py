from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict
from langchain_core.messages import HumanMessage, AIMessage
from loader import load_pdf_text, split_text
from rag import get_vectorstore, get_conversational_chain

app = FastAPI()

global_chain = None

class ChatRequest(BaseModel):
    question: str
    chat_history: List[Dict[str, str]] = []

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global global_chain
    text = load_pdf_text(file.file)
    chunks = split_text(text)
    vectorstore = get_vectorstore(chunks)
    global_chain = get_conversational_chain(vectorstore)
    return {"message": "PDF processed successfully"}

@app.post("/chat")
async def chat(request: ChatRequest):
    global global_chain
    if global_chain is None:
        return {"error": "Upload PDF first"}

    # Convert chat history to LangChain message objects
    history = []
    for msg in request.chat_history:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))

    response = global_chain.invoke({
        "question": request.question,
        "chat_history": history
    })

    return {"answer": response}

@app.post("/clear")
async def clear_chat():
    return {"message": "Chat cleared"}