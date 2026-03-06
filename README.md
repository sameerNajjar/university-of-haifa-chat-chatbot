# University of Haifa Chat Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for the **University of Haifa – Faculty of Computer & Information Science**.

The project scrapes and cleans faculty website content, builds a searchable embedding index, and serves a web-based chatbot that answers user questions using only retrieved university sources.

---

## Features

- **Domain-specific RAG pipeline** for the University of Haifa CIS website
- **Hybrid retrieval** that combines semantic embedding search with BM25-style keyword matching
- **Grounded answers with sources** returned from retrieved pages
- **Hebrew-focused responses** with support for multilingual content handling
- **FastAPI web app** with login, registration, chat creation, and stored chat history
- **SQLite persistence** for users, chats, and messages
- **Local LLM inference with Ollama**
- **Embedding-based indexing** using `intfloat/multilingual-e5-small`

---

## Project Structure

```text
university-of-haifa-chat-chatbot/
├── chatbot/
│   ├── analyze_logs.py
│   ├── hebrew_utils.py
│   ├── hybrid_retriever.py
│   ├── language_filter.py
│   ├── logger.py
│   ├── rag_chat_bot.py
│   ├── requirements.txt
│   ├── search_index.py
│   └── test.py
├── extract_data/
│   ├── build_index.py
│   ├── chunking.py
│   ├── clean_data.py
│   ├── extract_data.py
│   └── extract_pdfs_only.py
├── webapp/
│   ├── static/
│   ├── templates/
│   ├── __init__.py
│   ├── app.py
│   ├── db.py
│   └── rag_engine.py
└── README.md
```

---

## How It Works

### 1. Data extraction
The project crawls University of Haifa CIS sitemap pages and extracts page content into JSONL files.

Main script:

```bash
python ./extract_data/extract_data.py --out cis_pages.jsonl
```

It can also scan for linked PDF files and extract PDF text:

```bash
python ./extract_data/extract_pdfs_only.py --out cis_pdfs.jsonl
```

### 2. Cleaning
Raw extracted text is cleaned to remove repeated boilerplate, noisy short lines, and duplicated content.

```bash
python ./extract_data/clean_data.py --inp cis_pages.jsonl --out cis_pages_clean.jsonl
```

### 3. Chunking
Cleaned pages are split into overlapping chunks that are better suited for retrieval.

```bash
python ./extract_data/chunking.py --inp cis_pages_clean.jsonl --out cis_chunks.jsonl
```

### 4. Embedding + index building
Chunks are embedded with `intfloat/multilingual-e5-small`, then stored as:

- `cis_emb.npy` – dense embedding matrix
- `cis_meta.jsonl` – aligned metadata for each chunk

```bash
python ./extract_data/build_index.py \
  --inp cis_chunks.jsonl \
  --out_emb ./data/cis_emb.npy \
  --out_meta ./data/cis_meta.jsonl
```

### 5. Retrieval and answer generation
At query time, the system:

1. Encodes the user question
2. Runs hybrid retrieval over the indexed chunks
3. Selects top matching sources
4. Sends the sources to a local Ollama model
5. Returns an answer with source references

---

## Tech Stack

### Backend
- Python
- FastAPI
- SQLModel
- SQLite
- Jinja2

### Retrieval / NLP
- Sentence Transformers
- `intfloat/multilingual-e5-small`
- NumPy
- BM25 / hybrid retrieval

### LLM
- Ollama
- `qwen3:8b` by default

### Data Collection
- Requests
- Trafilatura
- BeautifulSoup
- XML sitemap parsing
- PDF extraction utilities

---

## Web Application

The web app provides:

- User registration and login
- Session-based authentication
- Multiple chat sessions per user
- Saved conversation history
- RAG-based answers from indexed University of Haifa data
- Source display for retrieved documents

Main entry point:

```bash
webapp/app.py
```

Important runtime files expected by the app:

```text
/data/cis_emb.npy
/data/cis_meta.jsonl
/data/app.db
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/sameerNajjar/university-of-haifa-chat-chatbot.git
cd university-of-haifa-chat-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Python dependencies

You can start with the package list already included in `chatbot/requirements.txt`:

```bash
pip install -r chatbot/requirements.txt
```

Depending on your setup and PDF pipeline usage, you may also need:

```bash
pip install trafilatura pypdf pdfplumber
```

### 4. Install and run Ollama

Install Ollama, then pull the model used by the app:

```bash
ollama pull qwen3:8b
```

Start Ollama locally so the API is available at:

```text
http://localhost:11434
```

---

## Environment Variables

The app uses the following environment variables:

```bash
CHAT_SECRET_KEY=change-this-secret
OLLAMA_URL=http://localhost:11434
LLM_MODEL=qwen3:8b
TOPK=5
NUM_CTX=8192
ALPHA=0.6
```

You can export them manually before running the server.

Linux / macOS:

```bash
export CHAT_SECRET_KEY="change-this-secret"
export OLLAMA_URL="http://localhost:11434"
export LLM_MODEL="qwen3:8b"
export TOPK=5
export NUM_CTX=8192
export ALPHA=0.6
```

Windows PowerShell:

```powershell
$env:CHAT_SECRET_KEY="change-this-secret"
$env:OLLAMA_URL="http://localhost:11434"
$env:LLM_MODEL="qwen3:8b"
$env:TOPK="5"
$env:NUM_CTX="8192"
$env:ALPHA="0.6"
```

---

## Running the Full Pipeline

### Step 1: Extract website content

```bash
python ./extract_data/extract_data.py --out cis_pages.jsonl
```

### Step 2: Clean extracted pages

```bash
python ./extract_data/clean_data.py --inp cis_pages.jsonl --out cis_pages_clean.jsonl
```

### Step 3: Chunk documents

```bash
python ./extract_data/chunking.py --inp cis_pages_clean.jsonl --out cis_chunks.jsonl
```

### Step 4: Build embeddings index

```bash
mkdir -p data
python ./extract_data/build_index.py \
  --inp cis_chunks.jsonl \
  --out_emb ./data/cis_emb.npy \
  --out_meta ./data/cis_meta.jsonl
```

### Step 5: Start the web app

```bash
uvicorn webapp.app:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

---

## CLI Search / Testing

You can inspect retrieval results directly with:

```bash
python ./chatbot/search_index.py \
  --emb ./data/cis_emb.npy \
  --meta ./data/cis_meta.jsonl \
  --query "מה תנאי הקבלה לתואר שני?" \
  --topk 5
```

You can also run the chatbot logic from the command line:

```bash
python ./chatbot/rag_chat_bot.py --llm qwen3:8b --topk 5
```

---

## Database Models

The web app stores data using SQLModel with three main tables:

- `User`
- `Chat`
- `Message`

This allows each registered user to maintain separate chat sessions and message history.

---

## Notes and Limitations

- The chatbot is only as good as the indexed source data.
- If a page is missing from the extraction pipeline, the chatbot may not answer correctly.
- Local inference quality depends on the Ollama model you use.

---

## Future Improvements

- Better multilingual query detection
- Improved source citation formatting in the UI
- Admin tools for refreshing the index
- Scheduled recrawling of university pages
- Docker support
- Better deployment documentation
- Evaluation scripts for retrieval quality and answer grounding

