set -x


python icl/eval_qwen3_icl.py -d 1 -i system -dt topk
python icl/eval_qwen3_icl.py -d 1 -i user -dt topk
python icl/eval_qwen3_icl.py -d 1 -i custom -dt topk

python icl/eval_qwen3_icl.py -d 3 -i system -dt topk
python icl/eval_qwen3_icl.py -d 3 -i user -dt topk
python icl/eval_qwen3_icl.py -d 3 -i custom -dt topk

python icl/eval_qwen3_icl.py -d 5 -i system -dt topk
python icl/eval_qwen3_icl.py -d 5 -i user -dt topk
python icl/eval_qwen3_icl.py -d 5 -i custom -dt topk
