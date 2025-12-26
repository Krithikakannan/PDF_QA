from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import subprocess

DB_PATH = r"C:\Users\Admin\Downloads\pdf\chroma_db"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.Client(
    Settings(
        persist_directory=DB_PATH,
        is_persistent=True
    )
)

# ⚠️ USE get_or_create_collection
collection = chroma_client.get_or_create_collection("pdf_chunks")

doc_count = collection.count()
print(f"📦 Number of documents in collection: {doc_count}")

if doc_count == 0:
    print("❌ No documents found. Run extract_pdf.py first.")
    exit()

query = input("\nAsk a question from the PDF: ")
query_embedding = embedding_model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

context = "\n\n".join(results["documents"][0])

prompt = f"""
You are a helpful assistant.
Answer ONLY using the context.

Context:
{context}

Question:
{query}

Answer:
"""
response = subprocess.run(
    ["ollama", "run", "llama3"],
    input=prompt,
    capture_output=True,
    encoding="utf-8",
    errors="ignore"
)


print("\n🧠 Answer:\n")
print(response.stdout)
