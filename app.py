import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import tempfile
import subprocess

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(page_title="PDF Q&A", layout="centered")

st.title("📄 PDF Question Answering System")
st.subheader("Upload a PDF and ask questions about the PDF.")

# -------------------------
# Load Embedding Model
# -------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_model()

# -------------------------
# ChromaDB Setup
# -------------------------
chroma_client = chromadb.Client(
    Settings(
        persist_directory="./chroma_db",
        is_persistent=True
    )
)

collection = chroma_client.get_or_create_collection(name="pdf_chunks")

# -------------------------
# PDF Upload
# -------------------------
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    st.success("✅ PDF uploaded successfully")

    with st.spinner("Processing PDF..."):

        # Save PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        # Extract text
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()

        # Chunking
        def split_text(text, chunk_size=800, overlap=100):
            chunks = []
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunks.append(text[start:end])
                start += chunk_size - overlap
            return chunks

        chunks = split_text(full_text)

        # Embeddings
        embeddings = embedding_model.encode(chunks, show_progress_bar=False)

        # Clear old data
        existing_ids = collection.get()["ids"]
        if existing_ids:
            collection.delete(ids=existing_ids)

        # Store new chunks
        collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            ids=[f"chunk_{i}" for i in range(len(chunks))]
        )

        st.success("🎉 PDF processed and stored successfully!")
        st.info(f"📦 Total chunks stored: {len(chunks)}")

# -------------------------
# Ask Questions Section
# -------------------------
st.divider()
st.header("❓ Ask a Question")

question = st.text_input("Type your question here")

if question:
    with st.spinner("Thinking... 🤔"):

        # Embed question
        query_embedding = embedding_model.encode(question).tolist()

        # Retrieve relevant chunks
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        context = "\n\n".join(results["documents"][0])

        # Prompt for LLaMA 3
        prompt = f"""
You are a helpful assistant.
Answer the question strictly using the context below.

Context:
{context}

Question:
{question}

Answer:
"""

        # Call Ollama
        response = subprocess.run(
            ["ollama", "run", "llama3"],
            input=prompt,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="ignore"
        )

        st.subheader("🧠 Answer")
        st.write(response.stdout)
