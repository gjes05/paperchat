# 🧠 PaperChat — RAG-Powered Document Q&A

PaperChat lets you upload PDF documents and have a grounded, conversational Q&A session with them — powered by LangChain, Gemini 2.5 Flash, Pinecone, and Streamlit.

---

## 🏗️ Architecture

```
PDF Upload → PyPDF Loader → Text Splitter (RecursiveCharacter)
    → Pinecone Llama Embeddings → Pinecone Vector Store
         ↓
User Question → Similarity Search (Top-K chunks)
    → Gemini 2.5 Flash + Conversation Memory → Grounded Answer + Sources
```

## 🗂️ Project Structure

```
PaperChat/
├── app.py              # Streamlit frontend
├── rag_pipeline.py     # Core RAG logic (ingest + query)
├── requirements.txt
├── .env.example
└── README.md
```

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/PaperChat.git
cd PaperChat
pip install -r requirements.txt
```

### 2. Set Up API Keys

```bash
cp .env.example .env
```

Edit `.env` with your keys:

| Variable | Where to get it |
|---|---|
| `GOOGLE_API_KEY` | [Google AI Studio](https://aistudio.google.com/app/apikey) |
| `PINECONE_API_KEY` | [Pinecone Console](https://app.pinecone.io/) |
| `PINECONE_INDEX_NAME` | Choose any name, e.g. `paperchat` |

### 3. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ✨ Features

- **Multi-document support** — upload and query across multiple PDFs simultaneously
- **Conversational memory** — follow-up questions use context from previous turns (sliding window of 5 turns)
- **Source citations** — every answer shows which file and page the information came from
- **Grounded answers** — the model is instructed to only answer from the document context
- **Clean UI** — dark-mode Streamlit interface with custom CSS

## 🔧 Configuration

Key parameters in `rag_pipeline.py`:

| Constant | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 1000 | Characters per chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `TOP_K` | 5 | Retrieved chunks per query |
| `CHAT_MODEL` | `gemini-2.5-flash` | LLM for answer generation |
| `EMBEDDING_MODEL` | `llama-text-embed-v2` | Pinecone embedding model |

## 📊 Evaluation Metrics (for extending the project)

- **Faithfulness**: Does the answer only use info from context? 
- **Answer Relevancy**: Is the answer relevant to the question?
- **Context Recall**: Are the right chunks being retrieved?

```bash
pip install ragas
```

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Gemini 2.5 Flash (Google) |
| Embeddings | Pinecone `llama-text-embed-v2` (1024-dim) |
| Vector DB | Pinecone (serverless) |
| Orchestration | LangChain |
| Frontend | Streamlit |
| PDF Parsing | PyPDF |