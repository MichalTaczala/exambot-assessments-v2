import os
import chromadb
from dotenv import load_dotenv

try:
    from src.ingest import get_openai_embedding
    from src.langsmith_logger import LangsmithLoggerWrapper
except ModuleNotFoundError:
    # Fallback for direct script execution
    from ingest import get_openai_embedding
    from langsmith_logger import LangsmithLoggerWrapper

load_dotenv()

PERSIST_DIR = os.path.abspath("data/chroma_db")


class RAGAgent:
    def __init__(self, persist_directory=PERSIST_DIR, collection_name="docs", logger=None, parent_run_id=None):

        self.client = chromadb.PersistentClient(path=persist_directory)
        collections = [c.name for c in self.client.list_collections()]
        if collection_name in collections:
            self.collection = self.client.get_collection(collection_name)
        else:
            self.collection = self.client.create_collection(collection_name)
        self.logger = logger if logger is not None else LangsmithLoggerWrapper()
        self.parent_run_id = parent_run_id

    def retrieve_context(self, question, top_k=3):
        try:
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
            # Log successful retrieval
            try:
                if self.parent_run_id:
                    self.logger.log_child_event(
                        name="RAGAgent.retrieve_context",
                        inputs={"question": question, "top_k": top_k},
                        outputs={"context": context},
                        tags=["RAG", "success"]
                    )
                else:
                    self.logger.log_event(
                        name="RAGAgent.retrieve_context",
                        inputs={"question": question, "top_k": top_k},
                        outputs={"context": context},
                        tags=["RAG", "success"]
                    )
            except Exception as log_err:
                print(f"[LangsmithLogger] Logging error: {log_err}")
            return context
        except Exception as e:
            # Log failure
            try:
                if self.parent_run_id:
                    self.logger.log_child_event(
                        name="RAGAgent.retrieve_context",
                        inputs={"question": question, "top_k": top_k},
                        outputs={"error": str(e)},
                        tags=["RAG", "failure"]
                    )
                else:
                    self.logger.log_event(
                        name="RAGAgent.retrieve_context",
                        inputs={"question": question, "top_k": top_k},
                        outputs={"error": str(e)},
                        tags=["RAG", "failure"]
                    )
            except Exception as log_err:
                print(f"[LangsmithLogger] Logging error: {log_err}")
            raise

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
