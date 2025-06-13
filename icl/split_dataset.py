import argparse
import math
import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_dataset
from transformers import \
    AutoTokenizer  # AutoModelForCausalLM is no longer directly used for inference
from vllm import LLM, SamplingParams


def calculate_perplexity_from_vllm_output(prompt: str, answer: str,
                                          vllm_output_item, tokenizer) -> float:
    """
    Calculates the perplexity of an answer from a single vLLM output item.

    This function extracts the log probabilities for the answer tokens from the
    vLLM output and computes the perplexity.

    Args:
        prompt (str): The original prompt string.
        answer (str): The original answer string.
        vllm_output_item (vllm.RequestOutput): A single output item from llm.generate().
        tokenizer: The tokenizer object used to encode prompt and answer.

    Returns:
        float: The calculated perplexity. Returns float('inf') if the answer
               contains no tokens or if log probabilities are unavailable.
    """
    # Encode prompt and answer separately to find their respective token lengths.
    # add_special_tokens=False ensures we count only the raw content tokens.
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)

    if not answer_ids:
        # If the answer is empty after tokenization, perplexity is infinite.
        return float('inf'), 0

    # Determine the starting index of the answer tokens in vLLM's processed sequence.
    # This assumes vLLM's internal tokenization aligns with the tokenizer's count.
    answer_start_idx = len(prompt_ids)

    # vLLM stores prompt_logprobs as a list of dictionaries. Each dictionary
    # contains the token ID and its log probability at that position.
    all_prompt_logprobs_list = vllm_output_item.prompt_logprobs

    if not all_prompt_logprobs_list or answer_start_idx >= len(all_prompt_logprobs_list):
        # Handle cases where vLLM didn't return logprobs or answer is out of bounds.
        # This can happen if the input was too short or there's an internal vLLM issue.
        print(f"Warning: Log probabilities missing or answer_start_idx out of bounds for request {vllm_output_item.request_id}. "
              f"Answer start: {answer_start_idx}, Total logprobs: {len(all_prompt_logprobs_list) if all_prompt_logprobs_list else 0}")
        return float('inf'), len(answer_ids)

    total_log_likelihood = 0.0
    num_scored_answer_tokens = 0

    # Iterate through the log probabilities starting from where the answer begins.
    for i in range(answer_start_idx, len(all_prompt_logprobs_list)):
        logprob_data = all_prompt_logprobs_list[i]
        if logprob_data:
            # Extract the single log probability value from the dictionary
            token_logprob = next(iter(logprob_data.values()))
            total_log_likelihood += token_logprob.logprob
            num_scored_answer_tokens += 1
        else:
            # This warning indicates a token position had no associated logprob, which is unusual.
            print(f"Warning: No logprob data found for token at index {i} in vLLM output {vllm_output_item.request_id}.")

    if num_scored_answer_tokens == 0:
        # If no actual answer tokens were processed, perplexity is infinite.
        return float('inf'), len(answer_ids)

    # Calculate the average negative log-likelihood over the scored answer tokens.
    average_negative_log_likelihood = -total_log_likelihood / num_scored_answer_tokens

    # Perplexity is e raised to the power of the average negative log-likelihood.
    # Handle potential floating point issues (e.g., extremely large NLL).
    if math.isinf(average_negative_log_likelihood) or math.isnan(average_negative_log_likelihood):
        return float('inf'), len(answer_ids)
    else:
        perplexity = math.exp(average_negative_log_likelihood)
        return perplexity, len(answer_ids)


def parse_args():
    parser = argparse.ArgumentParser(description="Split a dataset by PPL")
    parser.add_argument('--percent', '-p', type=float, default=None)
    parser.add_argument('--number', '-n', type=int, default=1000)
    parser.add_argument('--metrics', '-m', type=str, default='level')
    args = parser.parse_args()

    return args


def examine_dataframe(df, output_dir):
    outfile = os.path.join(output_dir, 'math_summation.txt')
    with open(outfile, 'w', encoding='utf-8') as f:
        print("\n--- Correlation between PPL and Response Length ---", file=f)
        # Pearson correlation coefficient between the two numerical columns
        print(f"Pearson Correlation (PPL vs Response Length): {df['ppl'].corr(df['response_length']):.2f}", file=f)

        print("\n--- Descriptive Statistics by Level ---", file=f)
        # Group by 'level' and calculate mean, median, and standard deviation
        difficulty_stats_ppl = df.groupby('level')['ppl'].agg(['mean', 'median', 'std']).sort_values('mean')
        difficulty_stats_response = df.groupby('level')['response_length'].agg(['mean', 'median', 'std']).sort_values('mean')

        print("\nAverage PPL by level:", file=f)
        print(difficulty_stats_ppl)
        print("\nAverage Response Length by level:", file=f)
        print(difficulty_stats_response)

    levels = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level ?']
    # 3. Visualize Relationships

    plt.style.use('seaborn-v0_8-darkgrid') # Set a nice plot style

    # Plot 1: Scatter plot of PPL vs. Response Length, colored by level
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x='response_length', y='ppl', hue='level',
                    palette='viridis', s=100, alpha=0.7, edgecolor='w')
    plt.title('PPL vs. Response Length by Level', fontsize=16)
    plt.xlabel('Response Length', fontsize=12)
    plt.ylabel('PPL (Perplexity)', fontsize=12)
    plt.legend(title='Level', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ppl_vs_response_length_scatter.png')) # Save the plot
    # plt.show()

    # Plot 2: Box plots for PPL across Level levels
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='level', y='ppl', palette='magma', order=levels)
    plt.title('PPL Distribution Across Level Levels', fontsize=16)
    plt.xlabel('Level', fontsize=12)
    plt.ylabel('PPL (Perplexity)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ppl_by_difficulty_boxplot.png')) # Save the plot
    # plt.show()

    # Plot 3: Box plots for Response Length across Level levels
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='level', y='response_length', palette='cividis', order=levels)
    plt.title('Response Length Distribution Across Level Levels', fontsize=16)
    plt.xlabel('Level', fontsize=12)
    plt.ylabel('Response Length', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'response_length_by_difficulty_boxplot.png')) # Save the plot
    # plt.show()

    # Plot 4: Pair Plot (useful for overall relationships if you have more columns)
    # This plot provides a quick overview of all pairwise relationships and distributions.
    # Be aware that for very large datasets, this might be slow.
    print("\n--- Generating Pair Plot (This might take a moment) ---")
    sns.pairplot(df, hue='level', palette='viridis', diag_kind='kde')
    plt.suptitle('Pair Plot of PPL, Response Length, and Level', y=1.02, fontsize=16)
    plt.savefig(os.path.join(output_dir, 'pair_plot.png')) # Save the plot


if __name__ == "__main__":
    args = parse_args()

    # --- Configuration ---
    data_dir = '/oss/public/user/liuts/datasets/math/MATH/competition_math/data/MATH/train'
    # data_dir = "/oss/public/user/liuts/datasets/math/AIME_2024"
    model_path = "/oss/public/user/liuts/model/Qwen3-8B"
    is_aime = 'AIME' in data_dir
        
    # --- Load Dataset ---
    print(f"Loading dataset from: {data_dir}")
    try:
        dataset = load_dataset(data_dir, 'default')['train']
        print(f"Dataset loaded. Contains {len(dataset)} examples.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1) # Exit if dataset cannot be loaded
    data_len = len(dataset)
    n = math.ceil(data_len * args.percent) if args.percent is not None else args.number
    outdir = 'icl/results'
    outfile = os.path.join(outdir, f'/Qwen3-8B-MATH-{args.metrics}-top{n}.parquet')

    # if os.path.exists(outfile):
    #     df = pd.read_parquet(outfile)
    #     examine_dataframe(df, outdir)
    #     print(df)
    #     exit()

    # --- Initialize vLLM and Tokenizer ---
    print(f"Initializing vLLM model: {model_path}")
    try:
        # Initialize vLLM with the specified model.
        # trust_remote_code=True is essential for many custom models like Qwen.
        # enforce_eager=True can sometimes help with stability.
        llm = LLM(model=model_path, trust_remote_code=True, enforce_eager=True)
        print("vLLM model initialized.")
    except Exception as e:
        print(f"Error initializing vLLM model: {e}. Ensure the model path is correct and enough VRAM is available.")
        exit(1)

    print(f"Loading tokenizer for model: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("Tokenizer loaded.")
    except Exception as e:
        print(f"Error loading tokenizer: {e}. Ensure the model path is correct.")
        exit(1)

    # --- Prepare Batch Inputs for vLLM ---
    problems = []
    solutions = []
    full_texts_for_vllm = [] # This list will hold the combined prompt+answer strings for vLLM

    print("Preparing data for batch inference...")

    for i, data in enumerate(dataset):
        if is_aime:
            problem = data['Problem']
            solution = data['Solution']
        else:
            problem = data['problem']
            solution = data['solution']

        # Concatenate problem and solution for vLLM input.
        # vLLM will process this combined string and give logprobs for all tokens.
        full_text = problem + solution
        full_texts_for_vllm.append(full_text)

        # Store original problem and solution to use when processing outputs
        problems.append(problem)
        solutions.append(solution)
    print(f"Prepared {len(full_texts_for_vllm)} inputs for batch inference.")

    # --- Define vLLM Sampling Parameters ---
    # max_tokens=1 is set to satisfy vLLM's requirement for generation.
    # prompt_logprobs=None means vLLM will return log probabilities for ALL input tokens.
    sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=1)

    # --- Perform Batch Inference with vLLM ---
    print("Starting batch inference with vLLM. This might take a while for large datasets...")
    start_time = time.time()
    try:
        # The core acceleration: passing all inputs to llm.generate() at once.
        vllm_outputs = llm.generate(full_texts_for_vllm, sampling_params=sampling_params)
    except Exception as e:
        print(f"Error during vLLM batch generation: {e}. Check your model and inputs.")
        exit(1)
    end_time = time.time()
    accumulated_time = end_time - start_time
    print(f"Batch inference completed for {len(vllm_outputs)} examples in {accumulated_time:.2f} seconds.")

    # --- Calculate Perplexities from vLLM Outputs ---
    data_list = []
    print("Calculating perplexities from vLLM outputs...")
    for i, output_item in enumerate(vllm_outputs):
        if dataset[i]['level'] == 'Level ?':
            continue
        # Retrieve the original prompt, solution, and level for this output item.
        original_problem = problems[i]
        original_solution = solutions[i]

        # Calculate perplexity using the dedicated function and vLLM's output.
        ppl, response_length = calculate_perplexity_from_vllm_output(
            original_problem,
            original_solution,
            output_item,
            tokenizer
        )
        data = dataset[i]
        data['ppl'] = ppl
        data['response_length'] = response_length
        data_list.append(data)
        print(f"Problem {i}: Perplexity = {ppl:.2f}")

    filtered_dataset = sorted(data_list, key=lambda x: x[args.metrics])[::-1]
    # --- Print Summary Results ---
    print(f"\n--- Summary ---")
    print(f"Total time elapsed (batch generation): {accumulated_time:.2f}s")
    df = pd.DataFrame(filtered_dataset)
    df = df[df['level'] != 'Level ?']
    df = df[:n]
    df.to_parquet(outfile, index=False)
    examine_dataframe(df, outdir)
    print(df)
