import os
import requests
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

try:
    from src.langsmith_logger import LangsmithLoggerWrapper
except ModuleNotFoundError:
    from langsmith_logger import LangsmithLoggerWrapper


class AssessorAgent:
    def __init__(self, api_key=OPENAI_API_KEY, model="gpt-4.1-nano", logger=None, parent_run_id=None):
        self.api_key = api_key
        self.model = model
        self.logger = logger if logger is not None else LangsmithLoggerWrapper()
        self.parent_run_id = parent_run_id

    def assess(self, question, context, answer):
        # Input validation
        if not all(isinstance(x, str) and x.strip() for x in [question, context, answer]):
            raise ValueError("All inputs must be non-empty strings.")
        prompt = (
            f"You are an exam assessor.\n"
            f"Question: {question}\n"
            f"Context: {context}\n"
            f"Student Answer: {answer}\n"
            f"\n"
            f"Evaluate the answer using the context.\n"
            f"Respond in JSON with two fields: score (integer 0-10) and feedback (string).\n"
            f"Example: {{\"score\": 7, \"feedback\": \"Missing details on X.\"}}\n"
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a strict but fair exam assessor."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 256
        }
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[ERROR] OpenAI API call failed: {e}")
            # Log failure
            try:
                if self.parent_run_id:
                    self.logger.log_child_event(
                        name="AssessorAgent.assess",
                        inputs={"question": question, "context": context, "answer": answer},
                        outputs={"error": str(e)},
                        tags=["Assessor", "failure"]
                    )
                else:
                    self.logger.log_event(
                        name="AssessorAgent.assess",
                        inputs={"question": question, "context": context, "answer": answer},
                        outputs={"error": str(e)},
                        tags=["Assessor", "failure"]
                    )
            except Exception as log_err:
                print(f"[LangsmithLogger] Logging error: {log_err}")
            return {"score": None, "feedback": "API error: could not assess answer."}
        # Parse JSON from model output
        import json
        try:
            result = json.loads(content)
            score = int(result["score"])
            feedback = str(result["feedback"])
            if not (0 <= score <= 10):
                raise ValueError("Score out of range.")
            # Log success
            try:
                if self.parent_run_id:
                    self.logger.log_child_event(
                        name="AssessorAgent.assess",
                        inputs={"question": question, "context": context, "answer": answer},
                        outputs={"score": score, "feedback": feedback},
                        tags=["Assessor", "success"]
                    )
                else:
                    self.logger.log_event(
                        name="AssessorAgent.assess",
                        inputs={"question": question, "context": context, "answer": answer},
                        outputs={"score": score, "feedback": feedback},
                        tags=["Assessor", "success"]
                    )
            except Exception as log_err:
                print(f"[LangsmithLogger] Logging error: {log_err}")
            return {"score": score, "feedback": feedback}
        except Exception as e:
            print(f"[ERROR] Could not parse model output: {e}\nModel output: {content}")
            # Log parse failure
            try:
                if self.parent_run_id:
                    self.logger.log_child_event(
                        name="AssessorAgent.assess",
                        inputs={"question": question, "context": context, "answer": answer},
                        outputs={"error": str(e), "model_output": content},
                        tags=["Assessor", "parse_failure"]
                    )
                else:
                    self.logger.log_event(
                        name="AssessorAgent.assess",
                        inputs={"question": question, "context": context, "answer": answer},
                        outputs={"error": str(e), "model_output": content},
                        tags=["Assessor", "parse_failure"]
                    )
            except Exception as log_err:
                print(f"[LangsmithLogger] Logging error: {log_err}")
            return {"score": None, "feedback": "Model output error."}


if __name__ == "__main__":
    agent = AssessorAgent()
    question = "What is Knowledge Representation and Reasoning?"
    context = "Knowledge Representation and Reasoning (KRR) is a field of AI focused on representing information about the world in a form that a computer system can utilize to solve complex tasks."
    answer = "It is about how computers can store and use knowledge to solve problems."
    result = agent.assess(question, context, answer)
    print("Assessment result:", result)
