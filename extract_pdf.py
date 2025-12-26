import fitz
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

DB_PATH = r"C:\Users\Admin\Downloads\pdf\chroma_db"
pdf_path = r"C:\Users\Admin\Downloads\sixthsem_elective syllabus.pdf"

doc = fitz.open(pdf_path)
full_text = ""
for page in doc:
    full_text += page.get_text()

def split_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+chunk_size])
        start += chunk_size - overlap
    return chunks

chunks = split_text(full_text)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(chunks, show_progress_bar=True)

chroma_client = chromadb.Client(
    Settings(
        persist_directory=DB_PATH,
        is_persistent=True
    )
)

collection = chroma_client.get_or_create_collection("pdf_chunks")

collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

print(f"✅ Stored {len(chunks)} chunks in ChromaDB")
