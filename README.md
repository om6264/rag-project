# 📄 Ask My Docs — Production RAG System

A domain-specific Question Answering system using **Hybrid Retrieval (BM25 + Vector Search)** with citation enforcement, built with LangChain, FAISS, and Groq LLaMA3.

## 🚀 Features
- **Hybrid Retrieval** — BM25 keyword search + FAISS vector search combined
- **Citation Enforcement** — Every answer cites the source page
- **FastAPI Backend** — Production-ready REST API
- **Clean UI** — Upload PDF and ask questions from browser
- **Free LLM** — Powered by Groq (LLaMA3-8b), no OpenAI cost

## 🛠 Tech Stack
- Python, FastAPI, LangChain
- FAISS (vector search), BM25 (keyword search)
- HuggingFace Embeddings (all-MiniLM-L6-v2)
- Groq API (LLaMA3-8b-8192)

## ⚙️ Setup

```bash
git clone https://github.com/yourusername/rag-project
cd rag-project
pip install -r requirements.txt
cp .env.example .env
# Add your GROQ_API_KEY to .env
uvicorn api:app --reload
```

Then open `index.html` in your browser.

## 📁 Project Structure
```
rag-project/
├── app.py           # Core RAG pipeline
├── api.py           # FastAPI backend
├── index.html       # Frontend UI
├── requirements.txt
├── vercel.json      # Vercel deployment config
└── docs/            # Put your PDF here
```

## 🏗 Architecture
```
PDF → Chunking → [FAISS Vector Index + BM25 Index]
                          ↓
            Query → Hybrid Retrieval → Top 5 Chunks
                          ↓
               LLaMA3 (Groq) → Answer with Citations
```
