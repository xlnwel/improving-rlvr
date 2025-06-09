import argparse
import gc
import glob
import json
import os
import random
import re
import time
from datetime import datetime

import numpy as np
import torch
from datasets import load_dataset
from scipy.special import comb
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import set_seed as hf_set_seed
from vllm import LLM, SamplingParams

INSTRUCTION_PROMPT = "Let's think step by step and output the final answer within \\boxed{}. "


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


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM Evaluation Script for 72B Model")

    parser.add_argument("--data_dir", type=str, default="/oss/public/user/liuts/datasets/math/AIME_2024")
    parser.add_argument("--instruction_type", "-i", type=str, default='user')
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--k", type=int, default=8)
    return parser.parse_args()


def prepare_data(data_dir):
    dataset = load_dataset(data_dir, 'default')

    dataset = dataset['train']

    return dataset


def save_res_jsonl(samples, save_path):
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "a", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("Saved to", save_path)


# units mainly from MathQA
unit_texts = [
    "east",
    "degree",
    "mph",
    "kmph",
    "ft",
    "m sqaure",
    " m east",
    "sq m",
    "deg",
    "mile",
    "q .",
    "monkey",
    "prime",
    "ratio",
    "profit of rs",
    "rd",
    "o",
    "gm",
    "p . m",
    "lb",
    "tile",
    "per",
    "dm",
    "lt",
    "gain",
    "ab",
    "way",
    "west",
    "a .",
    "b .",
    "c .",
    "d .",
    "e .",
    "f .",
    "g .",
    "h .",
    "t",
    "a",
    "h",
    "no change",
    "men",
    "soldier",
    "pie",
    "bc",
    "excess",
    "st",
    "inches",
    "noon",
    "percent",
    "by",
    "gal",
    "kmh",
    "c",
    "acre",
    "rise",
    "a . m",
    "th",
    "π r 2",
    "sq",
    "mark",
    "l",
    "toy",
    "coin",
    "sq . m",
    "gallon",
    "° f",
    "profit",
    "minw",
    "yr",
    "women",
    "feet",
    "am",
    "pm",
    "hr",
    "cu cm",
    "square",
    "v â € ™",
    "are",
    "rupee",
    "rounds",
    "cubic",
    "cc",
    "mtr",
    "s",
    "ohm",
    "number",
    "kmph",
    "day",
    "hour",
    "minute",
    "min",
    "second",
    "man",
    "woman",
    "sec",
    "cube",
    "mt",
    "sq inch",
    "mp",
    "∏ cm ³",
    "hectare",
    "more",
    "sec",
    "unit",
    "cu . m",
    "cm 2",
    "rs .",
    "rs",
    "kg",
    "g",
    "month",
    "km",
    "m",
    "cm",
    "mm",
    "apple",
    "liter",
    "loss",
    "yard",
    "pure",
    "year",
    "increase",
    "decrease",
    "d",
    "less",
    "Surface",
    "litre",
    "pi sq m",
    "s .",
    "metre",
    "meter",
    "inch",
]

unit_texts.extend([t + "s" for t in unit_texts])


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string


def convert_word_number(text: str) -> str:
    try:
        text = str(w2n.word_to_num(text))
    except:
        pass
    return text


def strip_string(string, skip_unit=False):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    # replace \\ with \
    string = string.replace("\\!", "")
    # string = string.replace("\\ ", "")
    # string = string.replace("\\\\", "\\")

    # matrix
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = (
        string.replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
    )

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("\\{", "{")
    string = string.replace("\\}", "}")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    if not skip_unit:
        # Remove unit: texts
        for _ in range(2):
            for unit_text in unit_texts:
                # use regex, the prefix should be either the start of the string or a non-alphanumeric character
                # the suffix should be either the end of the string or a non-alphanumeric character
                _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
                if _string != "":
                    string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace("\\(", "").replace("\\)", "")

    # convert word number to digit
    string = convert_word_number(string)

    # replace "\\text{...}" to "..."
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
        string = string.replace(key, "")
    string = string.replace("\\emptyset", r"{}")
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    # string = string.replace("\\cdot", "")
    if (
        string.startswith("{")
        and string.endswith("}")
        and string.isalnum()
        or string.startswith("(")
        and string.endswith(")")
        and string.isalnum()
        or string.startswith("[")
        and string.endswith("]")
        and string.isalnum()
    ):
        string = string[1:-1]

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace('"', "")

    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0*$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def extract_answer(pred_str, data_name='openr1-math-220k'):
    ans = pred_str.split("boxed")[-1]
    if len(ans) == 0:
        return ""
    elif ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    pred = a
    pred = re.sub(r"\n\s*", "", pred)
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = strip_string(pred, skip_unit=data_name in ["carp_en", "minerva_math"])
    return pred


def clean_gpu_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("CUDA cache cleared")
    
    for device in range(torch.cuda.device_count()):
        torch.cuda.set_device(device)
        torch.cuda.synchronize()


def eval_model(args, model_path, aime24_dataset, batch_size=100):
    k = args.k
    n = max(args.n, k)
    model_name = model_path.split('/')[-1]
    output_dir = f'icl/results/n={n}-k={k}/{model_name}-instruction_type={args.instruction_type}'

    if os.path.isdir(output_dir):
        pass
    else:
        os.makedirs(output_dir)

    llm = LLM(
        model=model_path,
        tensor_parallel_size=8,
        trust_remote_code=True,
        gpu_memory_utilization=0.6,
        dtype="bfloat16",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        top_k=20,
        max_tokens=32768,
        seed=args.seed, 
        n=n
    )
    for i in tqdm(range(0, len(aime24_dataset), batch_size)):
        start = i
        end = min(i + batch_size, len(aime24_dataset))

        problems, solutions, answers = None, None, None
        for name in ['Problem', 'problem']:
            if name in aime24_dataset.features:
                problems = aime24_dataset[name][start:end]
                break
        for name in ['Solution', 'solution']:
            if name in aime24_dataset.features:
                solutions = aime24_dataset[name][start:end]
                break
        for name in ['Answer', 'answer']:
            if name in aime24_dataset.features:
                answers = aime24_dataset[name][start:end]
                break
        
        # prompts = [f'Problem: {problem}\nAnswer: {INSTRUCTION_PROMPT}\n' for problem in problems]
        if args.instruction_type == 'system':
            messages = [
                [{"role": "system", "content": INSTRUCTION_PROMPT}, {"role": "user", "content": problem}]
                for problem in problems
            ]
            prompts = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        elif args.instruction_type == 'user':
            messages = [
                [{"role": "user", "content": f"{problem} {INSTRUCTION_PROMPT}"}]
                for problem in problems
            ]
            prompts = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        elif args.instruction_type == 'custom':
            prompts = [f'Problem: {problem}\nSolution: {INSTRUCTION_PROMPT}\n' for problem in problems]
        else:
            raise ValueError(f'Instruction type {args.instruction_type} not supported')

        outputs = llm.generate(prompts, sampling_params)
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        outputs_texts = [[o.text for o in output.outputs] for output in outputs]

        results = []
        for j, (output_texts, prompt, prob, solu, ans) in enumerate(zip(outputs_texts, prompts, problems, solutions, answers)):
            predictions = []
            c = 0
            for output_text in output_texts:
                pred_ans = extract_answer(output_text)
                predictions.append(pred_ans)
                if str(ans) == pred_ans:
                    c += 1

            result = {
                "prompt": prompt,
                "problem": prob,
                "solution": solu,
                "answer": ans,
                "responses": output_texts,
                "predictions": predictions, 
                "avg@k":  c / n,
                "pass@k": compute_pass_at_k(n, c, k),
            }
            results.append(result)

        out_file = f'{output_dir}/all_results.jsonl'
        save_res_jsonl(results, out_file)
        print(f"Saved {end} results to {out_file}")
    
    print("Evaluation finished. Results saved to:", out_file)
    del llm
    clean_gpu_memory()

def main():
    args = parse_args()
    set_seed(args.seed)
    
    aime24_dataset = prepare_data(args.data_dir)
    print('AIME Dataset size: ', len(aime24_dataset))

    batch_size = 100
    model_paths = glob.glob(f'/oss/public/user/liuts/model/Qwen3*', recursive=False)
    model_paths = [model_path for model_path in model_paths if '1.7B' not in model_path and '4B' not in model_path]
    model_paths = [model_path for model_path in model_paths if model_path.endswith('8B')]
    print('Model paths:', model_paths)

    for model_path in model_paths:
        eval_model(args, model_path, aime24_dataset, batch_size)


if __name__ == "__main__":
    main()
