import math
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


def calculate_perplexity(prompt, answer, llm, tokenizer) -> float:
    """
    Calculates the perplexity of a given answer in the context of a prompt using vLLM.

    Perplexity measures how well a probability model predicts a sample. A lower
    perplexity indicates a better model. This function computes the perplexity
    of the 'answer' part of the text, conditioned on the 'prompt' and preceding
    answer tokens.

    Args:
        model_path (str): The path or Hugging Face identifier for the Qwen model
                          (e.g., 'Qwen/Qwen1.5-8B-Chat').
        prompt (str): The input prompt string.
        answer (str): The answer string for which perplexity needs to be calculated.

    Returns:
        float: The calculated perplexity. Returns float('inf') if the answer
               contains no tokens.

    Raises:
        ValueError: If vLLM fails to return outputs or prompt log probabilities,
                    or if answer tokens are out of bounds.
        Exception: For general vLLM initialization or generation errors.
    """
    # Encode prompt and answer separately to find their respective token lengths.
    # We use add_special_tokens=False to count only the raw content tokens,
    # as vLLM's internal tokenization might handle special tokens differently
    # when processing the full string.
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)

    # Concatenate token IDs for the full sequence.
    # Note: If your model has a specific chat template, you might need to
    # apply it here before tokenizing to ensure correct token boundaries.
    # For a simple prompt+answer, direct concatenation is usually fine for perplexity.
    full_input_ids = prompt_ids + answer_ids

    # Convert to a PyTorch tensor and move to the device the model is on.
    # Add a batch dimension (unsqueeze(0))
    input_ids_tensor = torch.tensor([full_input_ids]).to(llm.device)

    # Calculate perplexity.
    # Perplexity for a sequence S is exp(-1/N * sum(log P(token_i | token_<i))),
    # where N is the number of tokens in S.
    # In practice, we calculate the negative log likelihood (NLL) and then exponentiate.
    # Hugging Face models can return logits. We'll calculate the loss.

    # The labels for perplexity calculation are essentially the input tokens shifted.
    # For a sequence [t1, t2, t3, t4], to predict t2 given t1, t3 given (t1, t2) etc.,
    # the labels should be [t2, t3, t4, -100]. -100 indicates ignored tokens.
    # We only want to calculate the loss over the 'answer' tokens.
    labels = input_ids_tensor.clone()
    # Mask out prompt tokens so they don't contribute to the loss calculation.
    # -100 is the default `ignore_index` for CrossEntropyLoss.
    labels[:, :len(prompt_ids)] = -100

    try:
        with torch.no_grad(): # No need to calculate gradients for perplexity
            outputs = llm(input_ids=input_ids_tensor, labels=labels)
            # outputs.loss is the average negative log-likelihood over the labeled tokens.
            # This is exactly what we need for perplexity.
            nll = outputs.loss.item() # Negative Log Likelihood
    except Exception as e:
        print(f"Error during model inference or loss calculation: {e}")
        return float('inf') # Return infinity on computation error

    if nll == 0.0 and len(answer_ids) > 0: # Handle cases where loss is exactly zero, which might happen with perfect prediction
        print("Warning: Loss is zero. This might indicate perfect prediction or an issue. Perplexity will be 1.0.")
        perplexity = 1.0
    elif nll == float('inf') or nll == float('-inf') or math.isnan(nll):
        print(f"Warning: Calculated NLL is {nll}. Perplexity will be infinite.")
        perplexity = float('inf')
    else:
        perplexity = math.exp(nll)

    return perplexity

if __name__ == "__main__":
    data_dir = '/oss/public/user/liuts/datasets/math/MATH/competition_math/data/MATH/train'
    dataset = load_dataset(data_dir, 'default')['train']
    model = "/oss/public/user/liuts/model/Qwen3-8B"

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.bfloat16, # Or torch.float16 or torch.float32 depending on hardware/preference
        device_map="auto", # Automatically maps model layers to available devices (GPU/CPU)
        trust_remote_code=True
    )
    llm.eval()

    accumulated_time = 0
    level_ppl = defaultdict(list)
    ppls = []    
    for i, data in enumerate(dataset):
        problem = data['problem']
        solution = data['solution']
        level = data['level']
        start = time.time()
        ppl = calculate_perplexity(problem, solution, llm, tokenizer)
        end = time.time()
        accumulated_time += end - start
        level_ppl['level'].append(ppl)
        ppls.append(ppl)
        print(f"Problem {i}: Difficulty = {level} Perplexity = {ppl:.2f}")
    
    print(f"Average Perplexity: {np.mean(ppls):.2f}")
    print(f"Time elapsed: {accumulated_time:.2f}s")