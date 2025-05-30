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
uv add python-dotenv pandas chromadb openai langsmith
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

### 5. Setting up file:

- Put knowledge base files(.txt or .pdf ) to the /data/knowledge_base folder. Presentations, notes etc

- Put student answers to the data/answers folder. it needs to be a csv file with student_id,question,answer columns

### 6. Running the program

```python
uv run main.py
```

then, the output file with scores will be generated into data/output/results.csv
