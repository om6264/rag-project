import os
import time
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
        for j in range(0, len(text), 600):
            chunk = text[j:j+600].strip()
            if len(chunk) > 50:
                chunks.append({"text": chunk, "page": i+1, "source": pdf_path})
    return chunks

def build_bm25_index(chunks):
    from rank_bm25 import BM25Okapi
    tokenized = [c["text"].lower().split() for c in chunks]
    return BM25Okapi(tokenized)

def retrieve(query, chunks, bm25, top_k=5):
    import numpy as np
    scores = bm25.get_scores(query.lower().split())
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]

def generate_answer(query, context_docs):
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    context = ""
    for i, doc in enumerate(context_docs):
        context += f"[Source {i+1} | Page {doc['page']}]\n{doc['text']}\n\n"

    prompt = f"""Answer ONLY based on the context below.
Cite sources like [Source X | Page Y] after every sentence.
If not found say: "I could not find this in the document."

Context:
{context}

Question: {query}
Answer:"""

    start = time.time()
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800
    )
    latency = round(time.time() - start, 2)
    return response.choices[0].message.content, latency

class RAGPipeline:
    def __init__(self, pdf_path: str):
        print("Loading PDF...")
        self.chunks = load_documents(pdf_path)
        print(f"Loaded {len(self.chunks)} chunks.")
        self.bm25 = build_bm25_index(self.chunks)
        print("Ready!")

    def query(self, question: str):
        docs = retrieve(question, self.chunks, self.bm25)
        answer, latency = generate_answer(question, docs)
        return {
            "question": question,
            "answer": answer,
            "latency_seconds": latency,
            "sources_used": len(docs)
        }
