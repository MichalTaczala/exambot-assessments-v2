# Assessment System

## Setup Instructions

### 1. Prerequisites
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (for fast dependency management)

### 2. Clone the Repository
```sh
git clone <your-repo-url>
cd exambot-assessments-v2
```
### 3. Environment Variables
Create a `.env` file in the project root with:
```
OPENAI_API_KEY=sk-...
LANGSMITH_API_KEY=...(optional)
LANGSMITH_PROJECT=...(optional)
LANGSMITH_ENDPOINT=...(optional)
LANGSMITH_TRACING=...(optional)
```

---

### 4. Setting up file:

- Put knowledge base files(.txt or .pdf ) to the /data/knowledge_base folder. Presentations, notes etc

- Put student answers to the data/answers folder. it needs to be a csv file with student_id,question,answer columns

### 5. Running the program(it should install all dependencies by itself)

```python
uv run main.py
```

then, the output file with scores will be generated into data/output/results.csv
