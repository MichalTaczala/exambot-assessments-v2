# Assessment System

A modular, agent-based assessment system for ingesting knowledge bases, evaluating student answers, and managing results with Python, FastAPI, and CSV storage.

## Features
- Knowledge base ingestion (PDF/TXT)
- RAG, Assessor, and Judge agents (OpenAI-powered)
- Langsmith logging integration
- Persistent result storage in CSV (no database required)
- REST API for result access and management (FastAPI)

---

## Setup Instructions

### 1. Prerequisites
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (for fast dependency management)

### 2. Clone the Repository
```sh
git clone <your-repo-url>
cd exambot-assessments-v2
```

### 3. Install Dependencies
```sh
uv uv add -r requirements.txt
# Or, for development:
uv add fastapi uvicorn python-dotenv pandas chromadb openai langsmith
```

### 4. Environment Variables
Create a `.env` file in the project root with:
```
OPENAI_API_KEY=sk-...
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=...
LANGSMITH_ENDPOINT=...
```

---

## Running the API Server

From the project root:
```sh
uvicorn src.api:app --reload --port 8000
```
- Visit [http://localhost:8000/docs](http://localhost:8000/docs) for interactive API docs (Swagger UI).

---

## API Endpoints

### List All Results
`GET /results`
- Optional query params: `student_id`, `question_id`

### Get Result by Answer ID
`GET /results/{answer_id}`

### Submit a New Result
`POST /results`
- JSON body (example):
```json
{
  "student_id": 1,
  "question_id": 101,
  "answer_id": 1001,
  "answer_text": "Sample answer",
  "score": 8,
  "feedback": "Good job!",
  "assessor": "AssessorAgent",
  "judge": "LLMAsAJudgeAgent",
  "timestamp": "2025-05-30T14:56:00.345386",
  "agent_logs": [
    {"agent": "RAG", "event": "retrieve", "status": "success", "timestamp": "2025-05-30T14:56:00.345139"}
  ]
}
```

---

## CSV Storage
- Results are stored in `src/results.csv`.
- Use `src/db_storage.py` to save/load/query results programmatically.

---

## Troubleshooting
- **ImportError (db_storage):** Ensure imports use `from src.db_storage import ...` when running from the project root.
- **Port already in use:** Stop other servers or use a different port with `--port`.
- **API not responding:** Check server logs for errors, ensure `.env` is set up, and dependencies are installed.

---

## Running Tests
- Run the main block in `src/db_storage.py` to test CSV storage logic:
```sh
python src/db_storage.py
```
- Use tools like `curl` or Postman to test API endpoints.

---

## Contributing & Contact
- PRs and issues welcome!
- Contact: [Your Name/Email]
