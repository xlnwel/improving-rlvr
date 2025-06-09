import pandas as pd

def load_jsonl_to_dataframe(file_path):
    return pd.read_json(file_path, lines=True)

# Usage
df = load_jsonl_to_dataframe('Algebra.jsonl')
print(f"DataFrame shape: {df.shape}")
print(df.head())
print(df.head()[''])