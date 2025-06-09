set -x


python icl/eval_qwen3_icl2.py -d 1 -i system
python icl/eval_qwen3_icl2.py -d 1 -i user
python icl/eval_qwen3_icl2.py -d 1 -i custom

python icl/eval_qwen3_icl2.py -d 3 -i system
python icl/eval_qwen3_icl2.py -d 3 -i user
python icl/eval_qwen3_icl2.py -d 3 -i custom

python icl/eval_qwen3_icl2.py -d 5 -i system
python icl/eval_qwen3_icl2.py -d 5 -i user
python icl/eval_qwen3_icl2.py -d 5 -i custom
