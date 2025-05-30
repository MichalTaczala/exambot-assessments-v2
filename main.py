import os
from dotenv import load_dotenv
from src.batch_assess import main as batch_assess
from src.ingest import embed_knowledge_base


def main():
    load_dotenv()
    required_keys = [
        'OPENAI_API_KEY',
        'LANGSMITH_API_KEY',
        'LANGSMITH_ENDPOINT',
        'LANGSMITH_PROJECT',
        'LANGSMITH_TRACING',
    ]
    missing = [k for k in required_keys if not os.getenv(k)]
    if missing:
        print(f"Error: Missing required environment variables: {', '.join(missing)}")
        exit(1)
    print("All required environment variables loaded.")
    print("Starting knowledge base ingestion...")
    # ingest()
    print("Knowledge base ingestion complete. Ready for further processing.")
    print("Hello from exambot-assessments-v2!")
    batch_assess()


if __name__ == "__main__":
    main()
