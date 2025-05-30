import os
import chromadb
from dotenv import load_dotenv

try:
    from src.ingest import get_openai_embedding
except ModuleNotFoundError:
    # Fallback for direct script execution
    from ingest import get_openai_embedding

load_dotenv()

PERSIST_DIR = os.path.abspath("chroma_db")


class RAGAgent:
    def __init__(self, persist_directory=PERSIST_DIR, collection_name="docs"):
        print(f"[DEBUG] RAGAgent ChromaDB persist directory: {persist_directory}")
        self.client = chromadb.PersistentClient(path=persist_directory)
        collections = [c.name for c in self.client.list_collections()]
        if collection_name in collections:
            self.collection = self.client.get_collection(collection_name)
        else:
            self.collection = self.client.create_collection(collection_name)

    def retrieve_context(self, question, top_k=3):
        embedding = get_openai_embedding(question)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        # Lower distance = more similar. Filter out empty docs.
        context = [
            (doc, meta, dist)
            for doc, meta, dist in zip(docs, metas, dists)
            if doc and doc.strip()
        ]
        return context

    def list_documents(self):
        # List all documents in the collection
        results = self.collection.get(include=["documents", "metadatas"])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        print(f"Collection contains {len(docs)} documents.")
        for i, (doc, meta) in enumerate(zip(docs, metas), 1):
            print(f"Doc {i}: {meta.get('filename', 'unknown')}, Snippet: {doc[:100]}...")


if __name__ == "__main__":
    agent = RAGAgent()
    agent.list_documents()
    question = "What is Knowledge Representation and Reasoning?"
    context = agent.retrieve_context(question, top_k=3)
    print(f"Top {len(context)} results for: '{question}'\n")
    for i, (doc, meta, dist) in enumerate(context, 1):
        print(f"Result {i} (distance={dist:.4f}):")
        print(f"Filename: {meta['filename']}")
        print(f"Snippet: {doc[:300]}\n...")
