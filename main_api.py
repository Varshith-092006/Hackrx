# main_api.py
import os
import fitz  # PyMuPDF
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import requests
import tempfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# Text cleaning

def clean_text(text):
    return text.encode("utf-8", "replace").decode("utf-8", "ignore")

# Download + extract text from PDF

def download_and_extract_text(url: str) -> str:
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to download: {url}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    doc = fitz.open(tmp_path)
    full_text = ""
    for page in doc:
        full_text += clean_text(page.get_text())
    return full_text

# Chunking

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size - overlap)
    ]

# FAISS indexing

def build_faiss_index(chunks, embed_model):
    vectors = embed_model.encode(chunks)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, vectors

# Search

def search_chunks(index, query, embed_model, all_chunks, top_k=3):
    q_vec = embed_model.encode([query])
    _, idx = index.search(q_vec, top_k)
    return [all_chunks[i] for i in idx[0]]

# Prompt

def build_prompt(context, query):
    return f"""
You are an intelligent assistant trained to interpret legal, insurance, or policy documents.
Below are the most relevant parts of the document based on the user's question.
--------------------
📄 Policy Snippets:
{context}
--------------------
❓ Question:
"{query}"
--------------------
📤 Output Instructions:
- Answer ONLY if the information is clearly present.
- If not mentioned, say "Not specified in the provided content".
- If partial, say "Partially mentioned" and explain why.
- Format:
```json
{{
  "answer": "<answer>",
  "clause_reference": "<clause>",
  "explanation": "<why this answer>"
}}
"""

# Query Gemini

def query_gemini(prompt):
    try:
        response = model.generate_content(prompt)
        return clean_text(response.text)
    except Exception as e:
        return f"Error: {str(e)}"

# Models

class RunRequest(BaseModel):
    documents: List[str]
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

# FastAPI app

app = FastAPI()

@app.post("/api/v1/hackrx/run", response_model=RunResponse)
def run_submission(req: RunRequest):
    try:
        all_chunks = []
        embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        for url in req.documents:
            text = download_and_extract_text(url)
            chunks = chunk_text(text)
            if chunks:
                all_chunks.extend(chunks)
        if not all_chunks:
            raise HTTPException(status_code=400, detail="No valid chunks found.")
        index, _ = build_faiss_index(all_chunks, embed_model)
        results = []
        for question in req.questions:
            top_chunks = search_chunks(index, question, embed_model, all_chunks)
            context = "\n---\n".join(top_chunks)
            prompt = build_prompt(context, question)
            result = query_gemini(prompt)
            results.append(result)
        return {"answers": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Render-compatible startup
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main_api:app", host="0.0.0.0", port=port, reload=False)