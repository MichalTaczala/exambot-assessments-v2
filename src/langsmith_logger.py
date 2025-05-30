from langsmith import Client
import os
from dotenv import load_dotenv
import uuid

load_dotenv()


LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")


class LangsmithLoggerWrapper:
    """
    Langsmith logger that supports parent/child run structure for unified logging.
    Usage:
        logger = LangsmithLoggerWrapper()
        logger.start_session(name="User Q&A", inputs={...})
        child = logger.log_child_event(name="RAG", inputs={...}, outputs={...})
        # ... repeat for other agents ...
        logger.complete_session(outputs={"final_answer": ...})
    """

    def __init__(self, api_key=None, project=None, endpoint=None):
        self.api_key = api_key or LANGSMITH_API_KEY
        self.project = project or LANGSMITH_PROJECT
        self.endpoint = endpoint or LANGSMITH_ENDPOINT
        if not self.api_key:
            raise ValueError("LANGSMITH_API_KEY not found in environment variables or .env file.")
        self.client = Client(api_key=self.api_key)
        self.parent_run_id = None
        self.child_run_ids = []

    def start_session(self, name, inputs=None, tags=None):
        """Start a parent run for a user/answer session."""
        run_id = str(uuid.uuid4())
        self.client.create_run(
            id=run_id,
            name=name,
            inputs=inputs or {},
            tags=tags or [],
            run_type="chain",
            project_name=self.project,
            api_url=self.endpoint
        )
        self.parent_run_id = run_id
        return run_id

    def log_child_event(self, name, inputs=None, outputs=None, tags=None):
        """
        Log a child run/event under the parent session and immediately mark it as completed.
        Returns the child run ID.
        """
        if not self.parent_run_id:
            raise ValueError("Parent run not started. Call start_session() first.")
        child_run_id = str(uuid.uuid4())
        self.client.create_run(
            id=child_run_id,
            name=name,
            inputs=inputs or {},
            outputs=outputs or {},
            tags=tags or [],
            run_type="tool",
            parent_run_id=self.parent_run_id,
            project_name=self.project,
            api_url=self.endpoint
        )
        self.child_run_ids.append(child_run_id)
        # Immediately mark as completed
        self.complete_run(child_run_id, outputs=outputs)
        return child_run_id

    def complete_run(self, run_id, outputs=None):
        # Mark a run (parent or child) as completed
        return self.client.update_run(
            run_id=run_id,
            outputs=outputs or {},
            status="completed"
        )

    def complete_session(self, outputs=None):
        # Mark the parent run as completed
        if not self.parent_run_id:
            raise ValueError("Parent run not started. Call start_session() first.")
        return self.complete_run(self.parent_run_id, outputs=outputs)

    def log_event(self, name, inputs=None, outputs=None, tags=None):
        # Backward compatible: log a single event as a standalone run, and mark as completed
        run_id = str(uuid.uuid4())
        self.client.create_run(
            id=run_id,
            name=name,
            inputs=inputs or {},
            outputs=outputs or {},
            tags=tags or [],
            run_type="tool",
            project_name=self.project,
            api_url=self.endpoint
        )
        self.complete_run(run_id, outputs=outputs)
        return run_id


if __name__ == "__main__":
    try:
        logger = LangsmithLoggerWrapper()
        result = logger.log_event(
            name="Test event from LangsmithLoggerWrapper",
            inputs={"foo": "bar"},
            outputs={"result": "success"},
            tags=["test"]
        )
        print("Log event sent. Result:", result)
    except Exception as e:
        print("[LangsmithLoggerWrapper] Error:", e)
