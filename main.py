from fastapi import FastAPI, UploadFile, File
from loader import load_pdf_text, split_text
from rag import get_vectorstore, get_conversational_chain

app = FastAPI()

global_chain = None

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global global_chain
    text = load_pdf_text(file.file)
    chunks = split_text(text)
    vectorstore = get_vectorstore(chunks)
    global_chain = get_conversational_chain(vectorstore)
    return {"message": "PDF processed successfully"}

@app.post("/chat")
async def chat(question: str):
    global global_chain
    if global_chain is None:
        return {"error": "Upload PDF first"}
    response = global_chain.invoke(question)
    return {"answer": response}