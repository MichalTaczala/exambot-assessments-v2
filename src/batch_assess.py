import os
import sys
import traceback
from datetime import datetime, timezone
import pandas as pd
import json
import threading
import warnings

# Import agents and storage
from src.agents.rag_agent import RAGAgent
from src.agents.assessor_agent import AssessorAgent
from src.agents.judge_agent import LLMAsAJudgeAgent
from src.ingest import embed_knowledge_base
from src.db_storage import save_result
from src.langsmith_logger import LangsmithLoggerWrapper

# Paths


def main():
    # ... all the current code ...
    ANSWERS_DIR = os.path.join(os.path.dirname(__file__), '../data', 'answers')
    ANSWERS_FILE = None
    for f in os.listdir(ANSWERS_DIR):
        if f.endswith('.csv'):
            ANSWERS_FILE = os.path.join(ANSWERS_DIR, f)
            break
    if not ANSWERS_FILE:
        print("No CSV file found in answers/. Exiting.")
        sys.exit(1)
    print("[1/3] Embedding knowledge base...")
    embed_knowledge_base()

    # 3. Load answers
    print("[2/3] Loading answers...")
    try:
        df = pd.read_csv(ANSWERS_FILE)
    except Exception as e:
        print(f"Failed to load answers: {e}")
        sys.exit(1)

    # 4. Run assessment workflow for each answer
    print("[3/3] Running assessment workflow...")
    logger = LangsmithLoggerWrapper()

    for idx, row in df.iterrows():
        try:
            student_id = row.get('student_id')
            question = row.get('question')
            answer = row.get('answer')
            # 1. Start parent run for this answer
            parent_run_id = logger.start_session(
                name=f"Assessment for student {student_id}, question {question[:30]}...",
                inputs={"student_id": student_id, "question": question, "answer": answer},
                tags=["assessment"]
            )
            # 2. Instantiate agents with logger and parent_run_id
            rag = RAGAgent(logger=logger, parent_run_id=parent_run_id)
            assessor = AssessorAgent(logger=logger, parent_run_id=parent_run_id)
            judge = LLMAsAJudgeAgent(logger=logger, parent_run_id=parent_run_id)
            # RAG: retrieve context
            context = rag.retrieve_context(question, top_k=3)
            context_text = '\n'.join([doc for doc, meta, dist in context])
            # Assessor: score and feedback
            attempt = 0
            while attempt < 3:
                assessment = assessor.assess(question, context_text, answer)
                score = assessment.get('score')
                feedback = assessment.get('feedback')
                # Judge: reasonableness
                judge_result = judge.is_reasonable(question, context_text, answer, score, feedback)
                is_reasonable = judge_result.get('reasonable')
                if is_reasonable:
                    break
                attempt += 1
            if attempt == 3:
                print(f"[ERROR] Failed to process row {idx}: {e}\n{traceback.format_exc()}")
                raise Exception(f"Failed to process row {idx}: {e}")
            result = {
                'student_id': student_id,
                'question_id': row.get('question_id', idx + 1),
                'question': question,
                'answer_id': row.get('answer_id', idx + 1),
                'answer_text': answer,
                'score': score,
                'feedback': feedback,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'agent_logs': [
                    {'agent': 'RAG', 'event': 'retrieve', 'status': 'success',
                        'timestamp': datetime.now(timezone.utc).isoformat()},
                    {'agent': 'Assessor', 'event': 'assess', 'status': 'success',
                        'timestamp': datetime.now(timezone.utc).isoformat()},
                    {'agent': 'Judge', 'event': 'judge', 'status': 'success', 'timestamp': datetime.now(timezone.utc).isoformat(),
                                                                                                            'result': judge_result}
                ]
            }
            save_result(result)
            # 3. Complete parent run in Langsmith
            logger.complete_session(outputs={
                "score": score,
                "feedback": feedback,
                "judge_result": judge_result
            })
            print(f"[OK] Student {student_id}, Question: {question[:30]}... => Score: {score}, Feedback: {feedback}")
        except Exception as e:
            print(f"[ERROR] Failed to process row {idx}: {e}\n{traceback.format_exc()}")
            continue

    print("\nBatch assessment complete. Results saved to src/results.csv.")

    # Wait for all non-daemon threads to finish (optional, usually not needed)
    for t in threading.enumerate():
        if t is not threading.main_thread() and t.is_alive():
            t.join(timeout=1)

    # Optionally, suppress all warnings at shutdown
    warnings.filterwarnings("ignore")


if __name__ == "__main__":
    print("[Assessment System] Starting batch assessment pipeline...")
    main()
