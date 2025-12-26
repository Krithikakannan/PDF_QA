# 📄 PDF Question Answering System (Offline)

This project is a local, offline PDF Question Answering system built using Retrieval-Augmented Generation (RAG). Users can upload a PDF, ask questions, and receive answers strictly grounded in the document content.

---

## 🚀 Features

- Upload and process PDF files
- Ask natural language questions
- Accurate answers using document context
- Fully offline (no internet required)
- Uses local LLM via Ollama (LLaMA 3)
- Web UI built with Streamlit
- Persistent vector storage using ChromaDB

---

## 🛠️ Tech Stack

- Python
- PyMuPDF (PDF extraction)
- SentenceTransformers (Embeddings)
- ChromaDB (Vector database)
- Ollama + LLaMA 3 (Local LLM)
- Streamlit (Web UI)

---

## 📂 Project Structure
.
├── extract_pdf.py # Extracts PDF and stores embeddings
├── ask_pdf.py # CLI-based question answering
├── app.py # Streamlit web application
├── chroma_db/ # Persistent vector database


---

## ⚙️ Setup Instructions

### 1. Install Dependencies
```bash
pip install streamlit chromadb sentence-transformers pymupdf
2. Install Ollama & LLaMA 3
ollama pull llama3

3. Run Streamlit App
streamlit run app.py

🧠 How It Works

Extracts text from PDF

Splits text into overlapping chunks

Generates embeddings

Stores embeddings in ChromaDB

Retrieves relevant chunks based on question

Sends context to LLaMA 3

Displays answer

📌 Notes

The system runs completely offline

Answers are generated strictly from PDF content

No data is sent to external servers

🔮 Future Enhancements

Chat history

Voice input

Multi-document support

User authentication

Mobile-friendly UI
