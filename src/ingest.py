import os
from PyPDF2 import PdfReader
import chromadb
from chromadb.utils import embedding_functions
import requests
from dotenv import load_dotenv

CUSTOM_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'custom_data')

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PERSIST_DIR = os.path.abspath("chroma_db")


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text() or ''
    return text


def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read()


def clean_text(text):
    # Simple cleaning: strip, normalize whitespace
    return ' '.join(text.strip().split())


def get_openai_embedding(text):
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "input": text,
        "model": "text-embedding-ada-002"
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()['data'][0]['embedding']


def ingest():
    print(f"[DEBUG] ChromaDB persist directory: {PERSIST_DIR}")
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    if "docs" in [c.name for c in client.list_collections()]:
        collection = client.get_collection("docs")
    else:
        collection = client.create_collection("docs")
    count = 0
    for fname in os.listdir(CUSTOM_DATA_DIR):
        fpath = os.path.join(CUSTOM_DATA_DIR, fname)
        if fname.lower().endswith('.pdf'):
            print(f'Extracting from PDF: {fname}')
            raw = extract_text_from_pdf(fpath)
            cleaned = clean_text(raw)
        elif fname.lower().endswith('.txt'):
            print(f'Extracting from TXT: {fname}')
            raw = extract_text_from_txt(fpath)
            cleaned = clean_text(raw)
        else:
            continue
        if cleaned:
            print(f'Vectorizing {fname}...')
            embedding = get_openai_embedding(cleaned[:8000])  # Truncate to avoid token limits
            collection.add(
                documents=[cleaned],
                embeddings=[embedding],
                metadatas=[{"filename": fname}],
                ids=[fname]
            )
            count += 1
            # Debug: print number of docs after each add
            docs = collection.get(include=["documents"])["documents"]
            print(f"[DEBUG] Collection now has {len(docs)} documents.")
    print(f"Ingestion complete. {count} documents vectorized and stored in ChromaDB.")
    if not os.path.exists(PERSIST_DIR):
        print(f"[WARNING] Persist directory {PERSIST_DIR} does not exist after ingestion!")
    else:
        print(f"[DEBUG] Persist directory {PERSIST_DIR} exists.")
    test_chromadb_query(collection)


def test_chromadb_query(collection):
    print("Testing ChromaDB query...")
    query = "Knowledge Representation and Reasoning introduction"
    # Use OpenAI to embed the query
    embedding = get_openai_embedding(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=1,
        include=["documents", "metadatas"]
    )
    print("Query results:")
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(f"Filename: {meta['filename']}")
        print(f"Document snippet: {doc[:300]}\n...")


if __name__ == "__main__":
    ingest()
