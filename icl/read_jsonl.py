import argparse
import glob
import json
import os

import pandas as pd


def read_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data

def pretty_print(output_path, data):
    with open(output_path, 'w', encoding='utf-8') as f:
        for k in ['problem', 'solution', 'answer']:
            print(f"{k}: {data[k]}", sep='\n', file=f)
            print('-'*100, file=f)
        
        for i, (r, p) in enumerate(zip(data['responses'], data['predictions'])):
            print(f'---{i}-th Entry---', file=f)
            # print(f"Response: {r}", sep='\n', file=f)
            # print('-'*100, file=f)
            print(f"Prediction: {p}", sep='\n', file=f)
            print('-'*100, file=f)
        
        print(f"Accuracy: {data['accuracy']}", file=f)
        print('-'*100, file=f)
        print(f"Pass@k: {data['pass@k']}", file=f)


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
    df = pd.DataFrame(columns=model_names)
    for file in files:
        data = read_jsonl(file)
        model_name = file.split("/")[3]
        model_name = model_name.replace('Qwen3-', '').replace('instruction_type=', '')

        for i, item in enumerate(data):
            assert item['avg@k'] is not None, item['avg@k']
            df.loc[i, model_name] = item['avg@k']

    df.loc['avg'] = df.mean()
    df.to_excel(f'{output_dir}/avg@k.xlsm', sheet_name='avg@k', index=False, engine='openpyxl')
    print(df)
