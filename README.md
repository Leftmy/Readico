# RAG Document Search & QA API

A FastAPI-based REST API service designed for document ingestion, semantic search using a vector database (Qdrant), result reranking, and LLM answer generation with exact source citations.

---

## 🛠 Tech Stack & Rationale

* **Python 3.11+ / FastAPI**: A high-performance asynchronous web framework providing automatic Swagger/OpenAPI documentation and robust data validation via Pydantic.
* **Qdrant (Vector Store)**: A specialized vector database optimized for fast semantic retrieval using vector embeddings. Supports rapid local in-memory execution (`:memory:`) as well as production-ready deployments.
* **Embedding & Reranker Service**: Implements a Two-Stage Retrieval pipeline. Vector search rapidly pulls candidate chunks, while the Reranker re-orders them based on semantic relevance, significantly improving context precision for the LLM.
* **LLM Service (OpenAI / Gemini/ Anthropic)**: Generates accurate answers strictly grounded in retrieved document snippets, providing complete source citations (filename, page number).
* **Pytest**: A testing framework used for unit and integration testing with dependency isolation via `unittest.mock`.

---

## 📦 Installation & Setup

1. **Clone the repository**:
```bash
git clone https://github.com/Leftmy/Readico
cd Readico

```


2. **Create and activate a virtual environment**:
```bash
python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

```


3. **Install dependencies from `pyproject.toml**`:
```bash
pip install --upgrade pip
pip install -e .

```


*(Or using Poetry: `poetry install`)*
4. **Configure environment variables**:
Create a `.env` file in the root directory:
```env
APP_NAME="RAG Search API"
ENVIRONMENT="development"
LLM_API_KEY="your-openai-api-key"
QDRANT_HOST="localhost"
QDRANT_PORT=6333
...
```

P.S. You can copy `.env.example` with following command:
```bash
# Linux/macOS:
cp .env.example .env
# Windows:
Copy .env.example .env

```



---

## 🚀 Running the Project

### Start the Local Development Server

Run the application using `uvicorn`:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

```

Once running, interactive Swagger UI documentation will be available at:
👉 **`[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)`**

---

## 🧪 Running Automated Tests

To run the complete test suite (including service unit tests and API integration tests):

```bash
pytest -v

```

To check code coverage:

```bash
pytest --cov=app --cov-report=term-missing

```

---

## 📡 API Usage & Examples

### 1. Health Check

* **HTTP Method**: `GET`
* **URL**: `/api/v1/health`

**cURL:**

```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/api/v1/health' \
  -H 'accept: application/json'

```

**Response (`200 OK`):**

```json
{
  "status": "ok",
  "app_name": "RAG Search API",
  "environment": "development"
}

```

---

### 2. Upload and Index a Document

* **HTTP Method**: `POST`
* **URL**: `/api/v1/upload`
* **Content-Type**: `multipart/form-data`

**cURL:**

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/api/v1/upload' \
  -F 'file=@/path/to/fastapi_doc.pdf;type=application/pdf'

```

**Response (`201 Created`):**

```json
{
  "id": "doc_12345",
  "filename": "fastapi_doc.pdf",
  "total_chunks": 12,
  "status": "indexed"
}

```

---

### 3. Search & Answer Generation (Chat)

* **HTTP Method**: `POST`
* **URL**: `/api/v1/chat`
* **Content-Type**: `application/json`

**cURL:**

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/api/v1/chat' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "How does FastAPI work?",
    "top_k": 3
  }'

```

**Response (`200 OK`):**

```json
{
  "query": "How does FastAPI work?",
  "answer": "FastAPI is a modern, high-performance web framework for building APIs with Python 3.8+ based on standard Python type hints.",
  "citations": [
    {
      "document_id": "doc_12345",
      "filename": "fastapi_doc.pdf",
      "page_number": 1,
      "snippet": "FastAPI enables building high performance API endpoints.",
      "relevance_score": 0.95
    }
  ]
}

```