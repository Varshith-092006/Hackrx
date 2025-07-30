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

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# 🧼 Sanitize text to avoid surrogate issues
def clean_text(text):
    return text.encode("utf-8", "replace").decode("utf-8", "ignore")

# 📄 Download and extract text from remote PDFs
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

# 🧩 Chunking logic
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size - overlap)
    ]

# 🧠 Build vector DB
def build_faiss_index(chunks, embed_model):
    vectors = embed_model.encode(chunks)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, vectors

# 🔎 Search top chunks
def search_chunks(index, query, embed_model, all_chunks, top_k=3):
    q_vec = embed_model.encode([query])
    _, idx = index.search(q_vec, top_k)
    return [all_chunks[i] for i in idx[0]]

# 📋 Prompt builder
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
- Read the policy snippets carefully.
- Answer ONLY if the information is clearly present.
- If the answer is not mentioned, say "Not specified in the provided content".
- If only partially available, say "Partially mentioned" and explain why.
- Return the result strictly in the following JSON format:

```json
{{
  "answer": "<short, clear answer or 'Not specified'>",
  "clause_reference": "<quote or summarize the most relevant clause or section>",
  "explanation": "<brief explanation of how the answer was derived>"
}}
"""

# 🤖 Gemini query
def query_gemini(prompt):
    try:
        response = model.generate_content(prompt)
        return clean_text(response.text)
    except Exception as e:
        return f"Error: {str(e)}"

# 🧾 FastAPI model classes
class RunRequest(BaseModel):
    documents: List[str]
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

# 🚀 FastAPI app
app = FastAPI()

@app.post("/hackrx/run", response_model=RunResponse)
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
            raise HTTPException(status_code=400, detail="No valid chunks found in any document")

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
