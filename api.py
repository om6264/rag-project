from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, shutil, time
from app import RAGPipeline

app = FastAPI(title="RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_pipeline = None

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"status": "RAG API is running"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global rag_pipeline
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    os.makedirs("docs", exist_ok=True)
    path = f"docs/{file.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    rag_pipeline = RAGPipeline(path)
    return {"message": f"PDF '{file.filename}' loaded successfully", "chunks": len(rag_pipeline.chunks)}

@app.post("/query")
def query(request: QueryRequest):
    global rag_pipeline
    if rag_pipeline is None:
        raise HTTPException(status_code=400, detail="No PDF uploaded yet. Please upload a PDF first.")

    result = rag_pipeline.query(request.question)
    return result

@app.get("/health")
def health():
    return {
        "status": "ok",
        "pdf_loaded": rag_pipeline is not None,
        "timestamp": time.time()
    }
