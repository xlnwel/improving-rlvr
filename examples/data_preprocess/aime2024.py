# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the DAPO-Math-17k dataset to multiturn format
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/aime2024")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_path = "Maxwell-Jia/AIME_2024"
    dataset = datasets.load_dataset(data_path, "default")

    train_dataset = dataset["train"]

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop("Problem")

            prompt = f'{problem} {instruction_following}'

            answer = example.pop("Answer")
            solution = example.pop("Solution")
            example.pop('ID')

            data = {
                "level": "Level 5", 
                "type": "Math", 
                "data_source": 'aime2024', 
                "prompt": [{"role": "user", "content": prompt}], 
                "ability": "math", 
                "reward_model": {"style": "rule", "ground_truth": str(answer)}, 
                "extra_info": {
                    "split": split, 
                    "index": idx, 
                    "raw_problem": problem, 
                    "solution": solution
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    save_path = os.path.join(local_dir, "test.parquet")
    train_dataset.to_parquet(save_path)
    print(f"{len(train_dataset)=}")
    print(f"Data saved to {save_path}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
