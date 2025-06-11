import argparse
import glob
import json
import os

import pandas as pd

from icl.eval_qwen3_icl2 import extract_answer


def read_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                line = json.loads(line)
                data.append(line)
    return data


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--output_dir", "-o", type=str, default="icl/results/n=8-k=8")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    output_dir = args.output_dir
    files = glob.glob(f"{output_dir}/Qwen3*/*jsonl")

    model_names = [f.split('/')[3] for f in files]
    # model_names = sorted(model_names, key=lambda x: float(x.split('-')[1][:-1]))
    model_names = sorted(model_names, key=lambda x: '-'.join(x.split('-')[1:-1]))[::-1]
    model_names = [m.replace('Qwen3-', '').replace('instruction_type=', '') for m in model_names]
    print('model names =', model_names)
    keys = ['prompt', 'problem', 'solution', 'answer', 'right_response', 'right_prediction', 'false_response', 'false_prediction']
    dfs = {k: pd.DataFrame(columns=model_names) for k in keys}

    for file in files:
        data = read_jsonl(file)
        filedir = os.path.dirname(file)
        for i, item in enumerate(data):
            content = '\n\n---\n---\n\n'.join([
                '\n\n'.join(['## Prompt', item['prompt'] if 'prompt' in item else item['problem']]), 
                '\n\n'.join(['## Solution', item['solution']]), 
                '\n\n'.join(['## Answer', str(item['answer'])]), 
            ])
            right_added = False
            false_added = False
            for (r, p) in zip(item['responses'], item['predictions']):
                if str(p) == str(item['answer']):
                    if right_added:
                        continue
                    content = '\n\n---\n---\n\n'.join([
                        content, 
                        '\n\n'.join(['## Right Response', r]), 
                        '\n\n'.join(['## Right Prediction', str(p)])
                    ])
                    right_added = True
                else:
                    if false_added:
                        continue
                    content = '\n\n---\n---\n\n'.join([
                        content, 
                        '\n\n'.join(['## False Response', r]), 
                        '\n\n'.join(['## False Prediction', str(p)])
                    ])
                    false_added = True
            with open(f'{filedir}/{i}.md', 'w', encoding='utf-8') as f:
                f.write(content)

