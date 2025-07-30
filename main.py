import os
import time
import fitz  # PyMuPDF
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# 📦 Load environment
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")  # Free-tier supported model

# 📄 Extract PDFs from folder
def extract_texts_from_folder(folder_path):
    texts = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            doc = fitz.open(os.path.join(folder_path, filename))
            text = "".join(page.get_text() for page in doc)
            texts[filename] = text
    return texts

# 🧩 Chunk text for embeddings
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size - overlap)
    ]

# 🧠 Build vector search index
def build_faiss_index(text_chunks, embed_model):
    vectors = embed_model.encode(text_chunks)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, vectors

# 🔍 Top-K retrieval
def search_index(index, query, embed_model, all_chunks, top_k=5):
    query_vector = embed_model.encode([query])
    _, indices = index.search(query_vector, top_k)
    return [all_chunks[i] for i in indices[0]]

# 🧾 Build Gemini prompt
def build_prompt(context, questions):
    questions_block = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    return f"""
You are an intelligent assistant trained to interpret legal, insurance, or policy documents.

--------------------
📄 Policy Snippets:
{context}
--------------------

❓ Questions:
{questions_block}

--------------------
📤 Instructions:
- Only answer based on the provided content.
- If a question is unanswered in the context, say "Not specified in the provided content".
- Format the output as a JSON list of answers:
[
  {{
    "answer": "...",
    "clause_reference": "...",
    "explanation": "..."
  }},
  ...
]
"""

# 🤖 Gemini query
def query_gemini(prompt, retries=3):
    for _ in range(retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            if "quota" in str(e).lower():
                print("⏳ Quota hit. Retrying in 60s...")
                time.sleep(60)
            else:
                raise RuntimeError(f"Gemini Error: {e}")
    raise RuntimeError("Gemini API failed after 3 retries.")

# 🚀 Main logic
if __name__ == "__main__":
    folder_path = "./bajaj_docs"
    documents = extract_texts_from_folder(folder_path)
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # 🔧 Chunk and index all documents
    all_chunks = []
    for filename, text in documents.items():
        chunks = chunk_text(text)
        print(f"📄 {filename}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"✅ Total Chunks: {len(all_chunks)}")

    if not all_chunks:
        raise RuntimeError("❌ No text chunks extracted!")

    index, _ = build_faiss_index(all_chunks, embed_model)

    # 🧪 Questions (customize here)
    questions = [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]

    # 🔍 Use top relevant context
    context_chunks = search_index(index, " ".join(questions), embed_model, all_chunks, top_k=5)
    context_text = "\n---\n".join(context_chunks)

    prompt = build_prompt(context_text, questions)
    result = query_gemini(prompt)

    print("\n📤 Gemini Output:\n")
    print(result)
