# app.py — updated: lower threshold, small-doc handling, summary-forcing
import os, time, traceback
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv
import numpy as np
import faiss

load_dotenv()

from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("all-MiniLM-L6-v2")

from groq import Groq
llm = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

app = FastAPI()

DOCUMENT_TEXT = ""
CHUNKS: List[str] = []
VECTORS = None
FAISS_INDEX = None

def chunk_text(text, size=400):
    # default chunk size increased a bit; but we'll override for very short docs below
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

def embed_chunks(chunks):
    print("[SERVER] embed_chunks() start: encoding", len(chunks), "chunks")
    vectors = embedder.encode(chunks, convert_to_numpy=True)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors.astype("float32") / norms.astype("float32")
    print("[SERVER] embed_chunks() done. shape:", vectors.shape)
    return vectors

def build_index(vectors):
    dim = vectors.shape[1]
    print("[SERVER] build_index() dim:", dim)
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    print("[SERVER] build_index() done. ntotal:", index.ntotal)
    return index

def retrieve_with_scores(query, top_k=3):
    global FAISS_INDEX, CHUNKS
    print("[SERVER] retrieve_with_scores() query:", (query[:120] + "...") if len(query)>120 else query)
    if FAISS_INDEX is None or len(CHUNKS) == 0:
        print("[SERVER] retrieve_with_scores() NO INDEX or NO CHUNKS")
        return [], []
    qvec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    qnorm = np.linalg.norm(qvec, axis=1, keepdims=True)
    qnorm[qnorm == 0] = 1.0
    qvec = qvec / qnorm
    k = min(top_k, int(FAISS_INDEX.ntotal))
    print(f"[SERVER] searching k={k} (ntotal={FAISS_INDEX.ntotal})")
    distances, indices = FAISS_INDEX.search(qvec, k)
    print("[SERVER] raw distances:", distances)
    sims = distances[0].tolist() if distances is not None else []
    idxs = indices[0].tolist() if indices is not None else []
    chunks = []
    scores = []
    for idx, score in zip(idxs, sims):
        if 0 <= idx < len(CHUNKS):
            chunks.append(CHUNKS[idx])
            scores.append(float(score))
    print(f"[SERVER] retrieve_with_scores() found {len(chunks)} chunks, scores: {scores}")
    return chunks, scores

def llm_answer(question):
    print("[SERVER] llm_answer() calling LLM for direct question")
    resp = llm.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}]
    )
    return resp.choices[0].message.content

def llm_rag_answer(question, chunks):
    print("[SERVER] llm_rag_answer() calling LLM with RAG context (chunks:", len(chunks),")")
    context = "\n\n".join(chunks)
    prompt = f"""
Use ONLY the context below to answer the question.

CONTEXT:
{context}

QUESTION:
{question}

If the answer is not in the context, say: 'Not available in document.'
"""
    resp = llm.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content

@app.post("/upload")
async def upload_pdf(file: Optional[UploadFile] = File(None), local_path: Optional[str] = Form(None)):
    import pypdf
    global DOCUMENT_TEXT, CHUNKS, VECTORS, FAISS_INDEX
    start_ts = time.time()
    print("[SERVER] /upload called. file:", bool(file), "local_path:", local_path)
    try:
        if file:
            pdf_reader = pypdf.PdfReader(file.file)
            text = ""
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text() or ""
                text += page_text + "\n"
                print(f"[SERVER] extracted page {i+1}: {len(page_text)} chars")
        elif local_path:
            print(f"[SERVER] using local_path: {local_path}")
            if not os.path.exists(local_path):
                print("[SERVER] local_path not found:", local_path)
                return JSONResponse(status_code=400, content={"status":"error","detail":"local_path not found on server"})
            pdf_reader = pypdf.PdfReader(local_path)
            text = ""
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text() or ""
                text += page_text + "\n"
                print(f"[SERVER] extracted page {i+1}: {len(page_text)} chars")
        else:
            print("[SERVER] upload called with no file and no local_path")
            return JSONResponse(status_code=400, content={"status":"error","detail":"No file or local_path provided."})

        DOCUMENT_TEXT = text.strip()

        # if document is small (few pages/chunks), treat full doc as single chunk for stronger similarity
        tentative_chunks = chunk_text(DOCUMENT_TEXT, size=400)
        if len(tentative_chunks) < 5:
            CHUNKS = [DOCUMENT_TEXT]
            print("[SERVER] small doc detected -> using full document as single chunk")
        else:
            CHUNKS = tentative_chunks
            print("[SERVER] chunked into", len(CHUNKS), "chunks")

        if len(CHUNKS) == 0:
            return JSONResponse(status_code=400, content={"status":"error","detail":"No text extracted from PDF (maybe scanned). Use OCR."})

        VECTORS = embed_chunks(CHUNKS)
        FAISS_INDEX = build_index(VECTORS)

        elapsed = time.time() - start_ts
        print(f"[SERVER] /upload finished OK in {elapsed:.2f}s")
        return {"status":"ok","chunks": len(CHUNKS)}
    except Exception as e:
        print("[SERVER] /upload exception:", e)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status":"error","detail":str(e)})

class Ask(BaseModel):
    question: str
    top_k: Optional[int] = 3
    similarity_threshold: Optional[float] = 0.05   # lowered default

@app.post("/ask")
async def ask_question(data: Ask):
    global FAISS_INDEX, CHUNKS
    question = data.question
    top_k = data.top_k or 3
    threshold = data.similarity_threshold if data.similarity_threshold is not None else 0.05

    print("[SERVER] /ask called. question:", (question[:120] + "...") if len(question)>120 else question)
    try:
        if FAISS_INDEX is None:
            print("[SERVER] No FAISS index found -> direct LLM")
            answer = llm_answer(question)
            return {"source":"llm_direct","answer":answer}

        # If user explicitly asked for a summary, force RAG using full doc context
        qlower = question.lower()
        if any(k in qlower for k in ["summary", "summar", "summarise", "summarize"]):
            print("[SERVER] summary-like question detected -> forcing RAG")
            answer = llm_rag_answer(question, CHUNKS)
            return {"source":"rag","answer":answer, "forced_summary": True}

        chunks, scores = retrieve_with_scores(question, top_k=top_k)
        best_score = max(scores) if scores else 0.0
        print("[SERVER] best_score:", best_score)

        if best_score >= threshold:
            print("[SERVER] Best score >= threshold -> using RAG")
            answer = llm_rag_answer(question, chunks)
            return {"source":"rag","answer":answer,"best_similarity":best_score,"chunks_used":chunks}
        else:
            print("[SERVER] Best score < threshold -> fallback to direct LLM")
            answer = llm_answer(question)
            return {"source":"llm_direct","answer":answer,"best_similarity":best_score}
    except Exception as e:
        print("[SERVER] /ask exception:", e)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status":"error","detail":str(e)})

@app.get("/status")
def status():
    ntotal = int(FAISS_INDEX.ntotal) if FAISS_INDEX is not None else 0
    return {"indexed": ntotal>0, "ntotal": ntotal}

@app.get("/")
def home():
    return {"message":"Day 8 RAG API (two-way) running - updated behavior"}
