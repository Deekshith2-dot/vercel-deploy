# api/app.py — Vercel-compatible version (lazy loading + no top-level crashes)
import os
import time
import traceback
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List

# ------------------- Lazy singletons (only import when actually used) -------------------
embedder = None
llm_client = None
faiss = None
np = None

def get_embedder():
    global embedder
    if embedder is None:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return embedder

def get_llm():
    global llm_client
    if llm_client is None:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in environment variables")
        llm_client = Groq(api_key=api_key)
    return llm_client

def get_faiss_and_np():
    global faiss, np
    if faiss is None or np is None:
        import faiss as _faiss
        import numpy as _np
        faiss = _faiss
        np = _np
    return faiss, np

# -----------------------------------------------------------------------------------
app = FastAPI()

# Global state
DOCUMENT_TEXT = ""
CHUNKS: List[str] = []
VECTORS = None
FAISS_INDEX = None

MODEL = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

def chunk_text(text, size=400):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

def embed_chunks(chunks):
    print("[SERVER] embed_chunks() start:", len(chunks), "chunks")
    vectors = get_embedder().encode(chunks, convert_to_numpy=True)
    faiss_mod, np_mod = get_faiss_and_np()
    norms = np_mod.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors.astype("float32") / norms.astype("float32")
    print("[SERVER] embed_chunks() done. shape:", vectors.shape)
    return vectors

def build_index(vectors):
    faiss_mod, _ = get_faiss_and_np()
    dim = vectors.shape[1]
    index = faiss_mod.IndexFlatIP(dim)
    index.add(vectors)
    print("[SERVER] build_index() done. ntotal:", index.ntotal)
    return index

def retrieve_with_scores(query, top_k=3):
    global FAISS_INDEX, CHUNKS
    if FAISS_INDEX is None or len(CHUNKS) == 0:
        return [], []
    qvec = get_embedder().encode([query], convert_to_numpy=True).astype("float32")
    faiss_mod, np_mod = get_faiss_and_np()
    qnorm = np_mod.linalg.norm(qvec, axis=1, keepdims=True)
    qnorm[qnorm == 0] = 1.0
    qvec = qvec / qnorm
    k = min(top_k, FAISS_INDEX.ntotal)
    distances, indices = FAISS_INDEX.search(qvec, k)
    sims = distances[0].tolist()
    idxs = indices[0].tolist()
    chunks_ret = [CHUNKS[i] for i in idxs if 0 <= i < len(CHUNKS)]
    scores_ret = [float(s) for i, s in zip(idxs, sims) if 0 <= i < len(CHUNKS)]
    return chunks_ret, scores_ret

def llm_answer(question):
    resp = get_llm().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}]
    )
    return resp.choices[0].message.content

def llm_rag_answer(question, chunks):
    context = "\n\n".join(chunks)
    prompt = f"""Use ONLY the context below to answer the question.

CONTEXT:
{context}

QUESTION:
{question}

If the answer is not in the context, say: 'Not available in document.'
"""
    resp = get_llm().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content


@app.get("/")
def home():
    return {"message": "Day-1-LLM RAG API running smoothly on Vercel!"}


@app.post("/upload")
async def upload_pdf(file: Optional[UploadFile] = File(None), local_path: Optional[str] = Form(None)):
    global DOCUMENT_TEXT, CHUNKS, VECTORS, FAISS_INDEX
    start_ts = time.time()
    try:
        import pypdf
        if file:
            pdf_reader = pypdf.PdfReader(file.file)
            text = "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
        elif local_path:
            if not os.path.exists(local_path):
                return JSONResponse(status_code=400, content={"status":"error","detail":"local_path not found"})
            pdf_reader = pypdf.PdfReader(local_path)
            text = "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
        else:
            return JSONResponse(status_code=400, content={"status":"error","detail":"No file or local_path"})

        DOCUMENT_TEXT = text.strip()
        tentative = chunk_text(DOCUMENT_TEXT, size=400)
        CHUNKS = [DOCUMENT_TEXT] if len(tentative) < 5 else tentative

        if not CHUNKS:
            return JSONResponse(status_code=400, content={"status":"error","detail":"No text extracted"})

        VECTORS = embed_chunks(CHUNKS)
        FAISS_INDEX = build_index(VECTORS)

        print(f"[SERVER] /upload OK in {time.time()-start_ts:.2f}s → {len(CHUNKS)} chunks")
        return {"status": "ok", "chunks": len(CHUNKS)}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status":"error","detail":str(e)})


class Ask(BaseModel):
    question: str
    top_k: Optional[int] = 3
    similarity_threshold: Optional[float] = 0.05

@app.post("/ask")
async def ask_question(data: Ask):
    global FAISS_INDEX, CHUNKS
    question = data.question.strip()
    top_k = data.top_k or 3
    threshold = data.similarity_threshold if data.similarity_threshold is not None else 0.05

    try:
        if FAISS_INDEX is None:
            answer = llm_answer(question)
            return {"source": "llm_direct", "answer": answer}

        # Force summary if keyword detected
        if any(k in question.lower() for k in ["summary", "summarize", "summarise"]):
            answer = llm_rag_answer(question, CHUNKS)
            return {"source": "rag", "answer": answer, "forced_summary": True}

        chunks, scores = retrieve_with_scores(question, top_k=top_k)
        best_score = max(scores) if scores else 0.0

        if best_score >= threshold:
            answer = llm_rag_answer(question, chunks)
            return {"source": "rag", "answer": answer, "best_similarity": best_score}
        else:
            answer = llm_answer(question)
            return {"source": "llm_direct", "answer": answer, "best_similarity": best_score}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status":"error","detail":str(e)})


@app.get("/status")
def status():
    n = int(FAISS_INDEX.ntotal) if FAISS_INDEX else 0
    return {"indexed": n > 0, "chunks": n}