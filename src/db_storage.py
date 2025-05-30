import csv
import json
import os
from datetime import datetime

CSV_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'output', 'results.csv')
FIELDNAMES = [
    'student_id', 'question_id', 'answer_id', 'answer_text',
    'score', 'feedback', 'assessor', 'judge', 'timestamp', 'agent_logs'
]


def save_result(result: dict):
    # Ensure all fields are present
    row = {k: result.get(k, '') for k in FIELDNAMES}
    # Serialize agent_logs if it's a dict or list
    if isinstance(row['agent_logs'], (dict, list)):
        row['agent_logs'] = json.dumps(row['agent_logs'])
    # Add timestamp if not present
    if not row['timestamp']:
        row['timestamp'] = datetime.utcnow().isoformat()
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_results():
    results = []
    if not os.path.isfile(CSV_FILE):
        return results
    with open(CSV_FILE, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Deserialize agent_logs if possible
            try:
                row['agent_logs'] = json.loads(row['agent_logs'])
            except Exception:
                pass
            results.append(row)
    return results


def get_results_by_student(student_id):
    return [r for r in load_results() if str(r['student_id']) == str(student_id)]


def get_results_by_question(question_id):
    return [r for r in load_results() if str(r['question_id']) == str(question_id)]


def get_result(answer_id):
    for r in load_results():
        if str(r['answer_id']) == str(answer_id):
            return r
    return None


if __name__ == "__main__":
    # Example usage
    test_result = {
        'student_id': 1,
        'question_id': 101,
        'answer_id': 1001,
        'answer_text': 'Sample answer',
        'score': 8,
        'feedback': 'Good job!',
        'assessor': 'AssessorAgent',
        'judge': 'LLMAsAJudgeAgent',
        'agent_logs': [{
            'agent': 'RAG', 'event': 'retrieve', 'status': 'success', 'timestamp': datetime.utcnow().isoformat()
        }]
    }
    save_result(test_result)
    print('Saved test result.')
    all_results = load_results()
    print('Loaded results:', all_results)
    print('Results for student 1:', get_results_by_student(1))
    print('Results for question 101:', get_results_by_question(101))
    print('Result for answer 1001:', get_result(1001))
