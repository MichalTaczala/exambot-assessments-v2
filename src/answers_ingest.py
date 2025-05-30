import os
import pandas as pd

ANSWERS_DIR = os.path.join(os.path.dirname(__file__), '..', 'answers')
REQUIRED_COLUMNS = ['student_id', 'question', 'answer']


def validate_row(row):
    return all(pd.notnull(row[col]) and str(row[col]).strip() != '' for col in REQUIRED_COLUMNS)


def ingest_answers():
    for fname in os.listdir(ANSWERS_DIR):
        if fname.lower().endswith('.csv'):
            fpath = os.path.join(ANSWERS_DIR, fname)
            print(f'Processing {fname}...')
            df = pd.read_csv(fpath)
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            if missing_cols:
                print(f'  ERROR: Missing columns: {missing_cols}')
                continue
            valid_mask = df.apply(validate_row, axis=1)
            valid = df[valid_mask]
            invalid = df[~valid_mask]
            print(f'  Valid rows: {len(valid)}')
            print(f'  Invalid rows: {len(invalid)}')
            if not invalid.empty:
                print(f'  Invalid row indices: {list(invalid.index)}')
            # For now, just print the first few valid rows
            print(valid.head())


if __name__ == "__main__":
    ingest_answers()
