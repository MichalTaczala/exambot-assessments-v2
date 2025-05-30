import os
import requests
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

try:
    from src.langsmith_logger import LangsmithLoggerWrapper
except ModuleNotFoundError:
    from langsmith_logger import LangsmithLoggerWrapper


class LLMAsAJudgeAgent:
    def __init__(self, api_key=OPENAI_API_KEY, model="gpt-4.1-nano", logger=None, parent_run_id=None):
        self.api_key = api_key
        self.model = model
        self.logger = logger if logger is not None else LangsmithLoggerWrapper()
        self.parent_run_id = parent_run_id

    def is_reasonable(self, question, context, answer, score, feedback):
        # Input validation
        if not all(isinstance(x, str) and x.strip() for x in [question, context, answer, feedback]):
            raise ValueError("All string inputs must be non-empty.")
        if not (isinstance(score, int) and 0 <= score <= 10):
            raise ValueError("Score must be an integer 0-10.")
        prompt = (
            f"You are an exam judge.\n"
            f"Question: {question}\n"
            f"Context: {context}\n"
            f"Student Answer: {answer}\n"
            f"Assessor Score: {score}\n"
            f"Assessor Feedback: {feedback}\n"
            f"\n"
            f"Is the score and feedback reasonable for the answer, given the context?\n"
            f"You don't care about the answer being correct or not, you only care about the score and feedback being reasonable for such an answer.\n"
            f"Respond in JSON: {{\"reasonable\": true/false, \"explanation\": \"...\"}}\n"
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a strict but fair exam judge."},
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
                        name="LLMAsAJudgeAgent.is_reasonable",
                        inputs={
                            "question": question,
                            "context": context,
                            "answer": answer,
                            "score": score,
                            "feedback": feedback},
                        outputs={"error": str(e)},
                        tags=["Judge", "failure"]
                    )
                else:
                    self.logger.log_event(
                        name="LLMAsAJudgeAgent.is_reasonable",
                        inputs={
                            "question": question,
                            "context": context,
                            "answer": answer,
                            "score": score,
                            "feedback": feedback},
                        outputs={"error": str(e)},
                        tags=["Judge", "failure"]
                    )
            except Exception as log_err:
                print(f"[LangsmithLogger] Logging error: {log_err}")
            return {"reasonable": None, "explanation": "API error: could not judge assessment."}
        # Parse JSON from model output
        import json
        try:
            result = json.loads(content)
            reasonable = bool(result["reasonable"])
            explanation = str(result["explanation"])
            # Log success
            try:
                if self.parent_run_id:
                    self.logger.log_child_event(
                        name="LLMAsAJudgeAgent.is_reasonable",
                        inputs={
                            "question": question,
                            "context": context,
                            "answer": answer,
                            "score": score,
                            "feedback": feedback},
                        outputs={"reasonable": reasonable, "explanation": explanation},
                        tags=["Judge", "success"]
                    )
                else:
                    self.logger.log_event(
                        name="LLMAsAJudgeAgent.is_reasonable",
                        inputs={
                            "question": question,
                            "context": context,
                            "answer": answer,
                            "score": score,
                            "feedback": feedback},
                        outputs={"reasonable": reasonable, "explanation": explanation},
                        tags=["Judge", "success"]
                    )
            except Exception as log_err:
                print(f"[LangsmithLogger] Logging error: {log_err}")
            return {"reasonable": reasonable, "explanation": explanation}
        except Exception as e:
            print(f"[ERROR] Could not parse model output: {e}\nModel output: {content}")
            # Log parse failure
            try:
                if self.parent_run_id:
                    self.logger.log_child_event(
                        name="LLMAsAJudgeAgent.is_reasonable",
                        inputs={
                            "question": question,
                            "context": context,
                            "answer": answer,
                            "score": score,
                            "feedback": feedback},
                        outputs={"error": str(e), "model_output": content},
                        tags=["Judge", "parse_failure"]
                    )
                else:
                    self.logger.log_event(
                        name="LLMAsAJudgeAgent.is_reasonable",
                        inputs={
                            "question": question,
                            "context": context,
                            "answer": answer,
                            "score": score,
                            "feedback": feedback},
                        outputs={"error": str(e), "model_output": content},
                        tags=["Judge", "parse_failure"]
                    )
            except Exception as log_err:
                print(f"[LangsmithLogger] Logging error: {log_err}")
            return {"reasonable": None, "explanation": "Model output error."}


if __name__ == "__main__":
    agent = LLMAsAJudgeAgent()
    question = "What is Knowledge Representation and Reasoning?"
    context = "Knowledge Representation and Reasoning (KRR) is a field of AI focused on representing information about the world in a form that a computer system can utilize to solve complex tasks."
    answer = "It is about how computers can store and use knowledge to solve problems."
    score = 8
    feedback = "Good explanation of the basic concept, but could be improved by mentioning that KRR involves not just storing knowledge but also reasoning with it to solve complex tasks."
    result = agent.is_reasonable(question, context, answer, score, feedback)
    print("Judge result:", result)
