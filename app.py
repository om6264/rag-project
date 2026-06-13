import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from rank_bm25 import BM25Okapi
from langchain.schema import Document
import numpy as np

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ─── Load and chunk PDF ───────────────────────────────────────────────────────
def load_documents(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    return chunks

# ─── Build FAISS vector index ─────────────────────────────────────────────────
def build_vector_index(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# ─── Build BM25 keyword index ─────────────────────────────────────────────────
def build_bm25_index(chunks):
    tokenized = [doc.page_content.lower().split() for doc in chunks]
    bm25 = BM25Okapi(tokenized)
    return bm25

# ─── Hybrid retrieval: BM25 + Vector ─────────────────────────────────────────
def hybrid_retrieve(query: str, vectorstore, bm25, chunks, top_k=5):
    # Vector search
    vector_results = vectorstore.similarity_search(query, k=top_k)
    vector_contents = set(doc.page_content for doc in vector_results)

    # BM25 search
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
    bm25_results = [chunks[i] for i in top_bm25_indices]

    # Merge results (deduplicate)
    combined = list(vector_results)
    for doc in bm25_results:
        if doc.page_content not in vector_contents:
            combined.append(doc)

    return combined[:top_k]

# ─── Generate answer with citations ──────────────────────────────────────────
def generate_answer(query: str, context_docs, llm):
    context = ""
    for i, doc in enumerate(context_docs):
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "document")
        context += f"[Source {i+1} | Page {page+1} | {source}]\n{doc.page_content}\n\n"

    prompt = f"""You are a helpful assistant. Answer ONLY based on the provided context.
Always cite your sources using [Source X | Page Y] format at the end of every sentence.
If the answer is not in the context, say "I could not find this in the document."

Context:
{context}

Question: {query}

Answer (with citations):"""

    start = time.time()
    response = llm.invoke(prompt)
    latency = round(time.time() - start, 2)

    return response.content, latency

# ─── Main RAG pipeline ────────────────────────────────────────────────────────
class RAGPipeline:
    def __init__(self, pdf_path: str):
        print("Loading documents...")
        self.chunks = load_documents(pdf_path)
        print(f"Loaded {len(self.chunks)} chunks.")

        print("Building vector index...")
        self.vectorstore = build_vector_index(self.chunks)

        print("Building BM25 index...")
        self.bm25 = build_bm25_index(self.chunks)

        print("Initializing LLM...")
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name="llama3-8b-8192"
        )
        print("RAG pipeline ready!\n")

    def query(self, question: str):
        docs = hybrid_retrieve(question, self.vectorstore, self.bm25, self.chunks)
        answer, latency = generate_answer(question, docs, self.llm)
        return {
            "question": question,
            "answer": answer,
            "latency_seconds": latency,
            "sources_used": len(docs)
        }

if __name__ == "__main__":
    rag = RAGPipeline("docs/document.pdf")
    while True:
        q = input("\nAsk a question (or type 'exit'): ")
        if q.lower() == "exit":
            break
        result = rag.query(q)
        print(f"\n📖 Answer:\n{result['answer']}")
        print(f"\n⏱ Latency: {result['latency_seconds']}s | Sources used: {result['sources_used']}")
