import argparse
import glob
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import set_seed as hf_set_seed
from vllm import LLM, SamplingParams

PROMPT_TEMPLATE = """Please act as an expert in mathematical problem-solving analysis. Your task is to distill the core methodology from a given math problem and its detailed solution.

Input: You will receive a math problem followed by its complete step-by-step solution.

Task:
Analyze the provided problem and its solution to extract the underlying general problem-solving methodology. Your summary should focus on:

The strategic approach: What high-level plan was employed to tackle this problem?

Key techniques/principles: What specific mathematical techniques, theorems, or concepts were utilized, and why were they appropriate for this problem type?

Reasoning progression: How did the solution logically flow from the problem statement to the final answer? What intermediate insights or transformations were crucial?

Generalizability: The most critical aspect is to describe the methodology in a way that is general enough to be applied to a class of similar mathematical problems, not just the specific one provided. Avoid mentioning specific numbers or problem-specific values unless they illustrate a general principle (e.g., "when dealing with inequalities involving absolute values," rather than "when solving ∣x−3∣<5").

Avoid:

Simply re-stating the steps of the solution.

Performing calculations or deriving the answer.

Output Format:
Present the distilled methodology as a concise, logically ordered list of steps or principles. Use bullet points or numbered lists. Each point should describe a generalizable action or reasoning process.

Here's the problem and its solution:

Problem: {problem}

Solution: {solution}

Your response should only contain the distilled methodology.
"""


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # For CUDA/cuDNN
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    hf_set_seed(seed) 
    print(f"Random seed set as {seed}")



def build_llm(model_path):
    llm = LLM(
        model=model_path,
        tensor_parallel_size=8,
        trust_remote_code=True,
        gpu_memory_utilization=0.6,
        max_model_len=32768, 
        dtype="bfloat16",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        top_k=20,
        max_tokens=32768,
        seed=42, 
        n=3
    )
    return llm, tokenizer, sampling_params


def build_prompts(problems, solutions):
    prompts = [PROMPT_TEMPLATE.format(problem=p, solution=s) for p, s in zip(problems, solutions)]
    return prompts


def summarize_methodology(llm, sampling_params, prompts, batch_size=200):
    methodologies = []
    for start in range(0, len(prompts), batch_size):
        end = start + batch_size
        sub_prompts = prompts[start:end]
        outputs = llm.generate(sub_prompts, sampling_params=sampling_params)
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        texts = [[o.text for o in output.outputs] for output in outputs]
        methodologies += texts
    return methodologies


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    model_paths = glob.glob(f'/oss/public/user/liuts/model/Qwen3-8B*', recursive=False)
    model_paths = [model_path for model_path in model_paths if '1.7B' not in model_path and '4B' not in model_path]
    model_paths = [model_path for model_path in model_paths if model_path.endswith('8B')]
    print('Model paths:', model_paths)

    dataset_path = '/oss/public/user/liuts/datasets/math/MATH/competition_math/data/MATH/train'
    dataset = load_dataset(dataset_path)['train']
    problems = [p for p in dataset['problem']]
    solutions = [s for s in dataset['solution']]

    model_methodologies = defaultdict(list)
    for model_path in model_paths:
        model_name = model_path.split('/')[-1]
        llm, tokenizer, sampling_params = build_llm(model_path)
        prompts = build_prompts(problems, solutions)
        methodologies = summarize_methodology(llm, sampling_params, prompts[:1])
        for p, s, m in zip(prompts, solutions, methodologies):
            print(f"Prompt: {p}")
            print(f"Solution: {s}")
            print(f"Methodology: {m}")
            print("-" * 100)
            row = {
                'problem': p,
                'solution': s,
                'methodology': m
            }
            model_methodologies[model_name].append(row)
    
    for model, method in model_methodologies.items():
        df = pd.DaataFrame(model_methodologies)
        df.to_parquet(f'icl/results/{model}_method.parquet', index=False)
        print(df)