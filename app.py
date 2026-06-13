import os
import time
import numpy as np
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def load_documents(pdf_path: str):
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    chunks = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        # Simple chunking: split every 500 chars
        for j in range(0, len(text), 500):
            chunk = text[j:j+500].strip()
            if chunk:
                chunks.append({"text": chunk, "page": i+1, "source": pdf_path})
    return chunks

def build_bm25_index(chunks):
    from rank_bm25 import BM25Okapi
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    return bm25

def build_vector_index(chunks):
    from sentence_transformers import SentenceTransformer
    import faiss
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, model

def hybrid_retrieve(query, chunks, bm25, faiss_index, embed_model, top_k=5):
    # BM25
    scores = bm25.get_scores(query.lower().split())
    bm25_top = np.argsort(scores)[::-1][:top_k]

    # Vector
    q_vec = embed_model.encode([query]).astype("float32")
    _, vec_top = faiss_index.search(q_vec, top_k)
    vec_top = vec_top[0]

    # Merge
    seen = set()
    results = []
    for i in list(bm25_top) + list(vec_top):
        if i not in seen:
            seen.add(i)
            results.append(chunks[i])
    return results[:top_k]

def generate_answer(query, context_docs):
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    context = ""
    for i, doc in enumerate(context_docs):
        context += f"[Source {i+1} | Page {doc['page']}]\n{doc['text']}\n\n"

    prompt = f"""You are a helpful assistant. Answer ONLY based on the context below.
Always cite sources like [Source X | Page Y] after every sentence.
If answer not found, say: "I could not find this in the document."

Context:
{context}

Question: {query}

Answer:"""

    start = time.time()
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    latency = round(time.time() - start, 2)
    return response.choices[0].message.content, latency

class RAGPipeline:
    def __init__(self, pdf_path: str):
        print("Loading PDF...")
        self.chunks = load_documents(pdf_path)
        print(f"Loaded {len(self.chunks)} chunks.")
        print("Building BM25 index...")
        self.bm25 = build_bm25_index(self.chunks)
        print("Building vector index...")
        self.faiss_index, self.embed_model = build_vector_index(self.chunks)
        print("Ready!\n")

    def query(self, question: str):
        docs = hybrid_retrieve(question, self.chunks, self.bm25, self.faiss_index, self.embed_model)
        answer, latency = generate_answer(question, docs)
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
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nLatency: {result['latency_seconds']}s")
