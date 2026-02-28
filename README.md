# 🚀 AI Engineering RAG Bot  
*A Retrieval-Augmented Generation (RAG) Application for Learning AI Concepts*

---

## 📌 Overview

This project is a **Retrieval-Augmented Generation (RAG) chatbot** designed to help users learn AI Engineering concepts from custom PDF documents.

Instead of relying only on pretrained knowledge, the system retrieves relevant information from a local knowledge base and generates structured educational responses using an LLM.

### Key Capabilities

- 📚 Context-aware answers from PDFs
- 🧠 Semantic search using embeddings
- ⚡ Fast inference using Groq LLM
- 🧾 Structured educational outputs
- 🌐 Interactive Streamlit interface

---

## 🏗️ Project Architecture
User Query
│
▼
Streamlit UI
│
▼
RAG Pipeline
├── Embedding Manager
├── Vector Database (ChromaDB)
├── Retriever
└── LLM (Groq - Llama 3.1)
│
▼
Structured Educational Response

---

## 📂 Project Structure
├── main_pipeline.py # Streamlit app + pipeline orchestration
├── rag.py # Retrieval logic
├── embedding_manager.py # Embedding generation
├── vector_db.py # ChromaDB vector storage
├── data/ # PDF knowledge base
├── vector_store/ # Persistent embeddings
├── .env # API keys
└── README.md

---

## ⚙️ Components

### 1️⃣ Embedding Manager
Generates semantic embeddings using SentenceTransformers.

- Model: `BAAI/bge-small-en-v1.5`
- Converts text into vector embeddings.

---

### 2️⃣ Vector Database (ChromaDB)

Stores embeddings persistently and enables similarity search.

Features:
- Automatic collection creation
- Metadata storage
- Fast retrieval

---

### 3️⃣ RAG Retriever

Responsible for:
- Query embedding generation
- Similarity search
- Threshold filtering
- Ranked document retrieval

---

### 4️⃣ Main Pipeline (Streamlit App)

Handles:
- PDF loading
- Text chunking
- Vector DB creation
- Prompt engineering
- Structured output generation

---

## 🧠 How It Works

1. PDFs are loaded from the `data/` directory.
2. Documents are split into chunks.
3. Chunks are converted into embeddings.
4. Embeddings are stored in ChromaDB.
5. User submits a query.
6. Relevant chunks are retrieved.
7. Context is sent to the LLM.
8. Structured educational response is generated.

---

## 🧾 Output Format

The chatbot returns structured learning content:

- ✅ Definition
- ✅ 3 Pros
- ✅ 3 Cons
- ✅ Use Case Explanation

---

## 🔧 Installation

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd rag-ai-bot
2. Create Virtual Environment
python -m venv .venv

Activate environment:

Windows

.venv\Scripts\activate

Mac/Linux

source .venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt

Example dependencies:

streamlit
langchain
langchain-community
langchain-groq
chromadb
sentence-transformers
pymupdf
python-dotenv
tqdm
pydantic
4. Environment Variables

Create a .env file:

GROQ_API_KEY=your_api_key_here
▶️ Run Application
streamlit run main_pipeline.py

Open browser:

http://localhost:8501
📊 Features

Retrieval-Augmented Generation

Persistent Vector Database

Structured LLM Outputs

PDF Knowledge Base

Semantic Search

Streamlit UI

Educational AI Assistant

🔮 Future Improvements

Multi-document upload UI

Hybrid search (keyword + semantic)

Conversation memory

FastAPI deployment

Docker support

Model switching

Evaluation dashboard

🧑‍💻 Tech Stack

Python

LangChain

ChromaDB

Sentence Transformers

Groq (Llama 3.1)

Streamlit

PyMuPDF

🤝 Contribution

Contributions are welcome!

Fork repository

Create feature branch

Commit changes

Open Pull Request

📜 License

MIT License — free to use and modify.

⭐ Acknowledgement

Built as part of an AI Engineering learning journey focused on practical RAG system design and LLM integration.


---
