import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import numpy as np
import os
import traceback
import time

from dotenv import load_dotenv

load_dotenv()

def safe_print(msg):
    print(msg, flush=True)

safe_print("== RAG DEBUG START ==")

# ---------- Basic env checks ----------
safe_print("Checking environment variables...")
api_key = os.environ.get("GROQ_API_KEY")
model_env = os.environ.get("MODEL_NAME")
safe_print(f"GROQ_API_KEY present: {bool(api_key)}")
safe_print(f"MODEL_NAME (env): {model_env}")

if not api_key:
    safe_print("❌ Missing GROQ_API_KEY.")
    raise SystemExit(1)

MODEL = model_env or "llama-3.1-8b-instant"

# ---------- Load embedding model ----------
safe_print("Loading embedding model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
safe_print("Embedding model loaded. Dim: " + str(embed_model.get_sentence_embedding_dimension()))

# ---------- Init LLM client ----------
safe_print("Initializing Groq client...")
llm = Groq(api_key=api_key)
safe_print("Groq client initialized.")

# ------------------------
# Chunk function
# ------------------------
def chunk_text(text, chunk_size=300):
    safe_print(f"Chunking text into chunks of ~{chunk_size} words...")
    chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    safe_print(f"Created {len(chunks)} chunks.")
    return chunks

# ------------------------
# Embed function
# ------------------------
def embed_chunks(chunks):
    safe_print("Creating embeddings...")
    vectors = embed_model.encode(chunks)
    vectors = np.array(vectors).astype('float32')
    safe_print("Embeddings shape: " + str(vectors.shape))
    return vectors

# ------------------------
# Create FAISS index
# ------------------------
def create_faiss_index(vectors):
    safe_print("Creating FAISS index...")
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    safe_print(f"FAISS index created. Total vectors: {index.ntotal}")
    return index

# ------------------------
# Search similar chunks
# ------------------------
def search(query, index, chunks, top_k=2):
    safe_print(f"Searching top_k={top_k} ...")
    qvec = embed_model.encode([query]).astype('float32')
    distances, indices = index.search(qvec, top_k)
    safe_print("Distances: " + str(distances))
    return [chunks[i] for i in indices[0]]

# ------------------------
# LLM answer
# ------------------------
def answer_with_context(question, chunks):
    safe_print("Preparing prompt...")
    context = "\n\n".join(chunks)
    prompt = f"""
CONTEXT:
{context}

QUESTION:
{question}

Answer ONLY from context.
"""

    safe_print("Calling LLM...")
    resp = llm.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content

# ------------------------
# RAG Pipeline
# ------------------------
def rag(query, document):
    safe_print("Starting RAG...")
    chunks = chunk_text(document)
    vectors = embed_chunks(chunks)
    index = create_faiss_index(vectors)
    top_chunks = search(query, index, chunks)
    safe_print("Top chunks found.")
    answer = answer_with_context(query, top_chunks)
    return answer

# ------------------------
# Test
# ------------------------
if __name__ == "__main__":
    doc = """
    Flutter is a UI toolkit created by Google.
    It is used to build cross platform mobile apps.
    React Native is another mobile framework.
    Virat Kohli is an Indian cricketer.
    Flutter uses Dart language.
    """
    question = "What does Flutter use?"
    out = rag(question, doc)
    safe_print("\nRAG Final Answer:\n" + out)
    safe_print("== RAG DEBUG END ==")
