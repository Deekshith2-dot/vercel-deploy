import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

from dotenv import load_dotenv

load_dotenv()

# -------------------------
# Load embedding & LLM
# -------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

API_KEY = os.environ.get("GROQ_API_KEY")
MODEL = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")

if not API_KEY:
    raise ValueError("Set GROQ_API_KEY first")

llm = Groq(api_key=API_KEY)


# -------------------------
# 1. Extract text from PDF
# -------------------------
def extract_pdf_text(path):
    # Now we read from OCR text file instead of the actual PDF
    return open("resume_text.txt", "r", encoding="utf-8").read()


# -------------------------
# 2. Chunk text
# -------------------------
def chunk_text(text, chunk_size=400):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks


# -------------------------
# 3. Embeddings
# -------------------------
def embed_chunks(chunks):
    vectors = embed_model.encode(chunks)
    return np.array(vectors).astype("float32")


# -------------------------
# 4. Build FAISS index
# -------------------------
def build_faiss(vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index


# -------------------------
# 5. Search FAISS
# -------------------------
def search_chunks(query, index, chunks, top_k=3):
    qv = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(qv, top_k)

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])
    return results


# -------------------------
# 6. Ask LLM with context
# -------------------------
def ask_llm(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""
Use the following context to answer the question:

CONTEXT:
{context}

QUESTION:
{question}

Give a clear and short answer based ONLY on the context.
"""

    response = llm.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content


# -------------------------
# 7. Full CHAT-WITH-PDF function
# -------------------------
def chat_with_pdf(pdf_path, question):
    print("Extracting PDF...")
    text = extract_pdf_text(pdf_path)

    print("Chunking...")
    chunks = chunk_text(text)

    print("Embedding chunks...")
    vectors = embed_chunks(chunks)

    print("Building FAISS index...")
    index = build_faiss(vectors)

    print("Searching similar chunks...")
    top_chunks = search_chunks(question, index, chunks)

    print("Asking LLM...")
    answer = ask_llm(question, top_chunks)
    return answer


# -------------------------
# Test
# -------------------------
if __name__ == "__main__":
    pdf_file = "sample.pdf"  # replace with your PDF file name
    question = "What is the summary of this document?"
    print(chat_with_pdf(pdf_file, question))
