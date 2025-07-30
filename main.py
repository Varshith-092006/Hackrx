import os
import fitz  # PyMuPDF
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import time

# Load environment variables
load_dotenv()

# ✅ Configure Gemini with your API Key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")  # Use free tier supported model

# ✅ Extract text from PDFs in a folder
def extract_texts_from_folder(folder_path):
    texts = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            doc = fitz.open(os.path.join(folder_path, filename))
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            texts[filename] = full_text
    return texts

# ✅ Split long text into overlapping chunks
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size - overlap)
    ]

# ✅ Build FAISS index
def build_faiss_index(text_chunks, embed_model):
    if not text_chunks:
        raise ValueError("No text chunks provided to build the index.")
    vectors = embed_model.encode(text_chunks)
    if len(vectors) == 0 or len(vectors.shape) < 2:
        raise ValueError("Embedding failed: No dimension in the encoded vectors.")
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index, vectors

# ✅ Semantic search for relevant chunks
def search_index(index, query, embed_model, all_chunks, top_k=2):
    query_vector = embed_model.encode([query])
    _, indices = index.search(query_vector, top_k)
    return [all_chunks[i] for i in indices[0]]

# ✅ Prompt builder
def build_prompt(context, questions):
    questions_block = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    return f"""
You are an intelligent assistant trained to interpret legal, insurance, or policy documents.
Below are the most relevant parts of the document based on the user's question.

--------------------
\ud83d\udcc4 Policy Snippets:
{context}
--------------------

\ud83e\udd14 Questions:
{questions_block}

--------------------
\ud83d\udce4 Output Instructions:
- Answer ONLY if the information is clearly present.
- If not mentioned, say "Not specified in the provided content".
- Format your answer as JSON list of answers.
"""

# ✅ Gemini Query Handler with Retry
def query_gemini(prompt, retries=3):
    for i in range(retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            if "quota" in str(e).lower():
                print(f"\ud83d\udd52 Quota limit hit. Retrying in 60s...")
                time.sleep(60)
            else:
                raise
    raise RuntimeError("\u274c Gemini API failed after retries.")

# ✅ Main driver
if __name__ == "__main__":
    folder_path = "./bajaj_docs"
    documents = extract_texts_from_folder(folder_path)
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    all_chunks = []
    for filename, text in documents.items():
        chunks = chunk_text(text)
        print(f"\ud83d\udcc4 {filename} \u2192 {len(chunks)} chunks")
        if chunks:
            all_chunks.extend(chunks)

    print(f"\n\u2705 Total Chunks Collected: {len(all_chunks)}")

    if not all_chunks:
        raise RuntimeError("\u274c No valid text chunks found.")

    index, _ = build_faiss_index(all_chunks, embed_model)

    # ✅ Batch user questions into a single query
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

    # ✅ Use top relevant context
    combined_context = search_index(index, "health insurance policy benefits and limits", embed_model, all_chunks, top_k=5)
    context_text = "\n---\n".join(combined_context)
    prompt = build_prompt(context_text, questions)

    # ✅ Get Gemini response
    result = query_gemini(prompt)
    print("\n\ud83d\udd0d Gemini Output:\n")
    print(result)