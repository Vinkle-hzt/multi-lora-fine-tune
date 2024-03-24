python test.py \
  --base_model /home/hzt/model/llama-7b-hf \
  --config ./config/dummy_pp.json \
  --pipeline \
  --rank 0 \
  --device cuda:0 \
  --balance 12 12 11

python  test.py \
  --base_model /home/hzt/model/llama-7b-hf \
  --config ./config/dummy_pp.json \
  --pipeline \
  --rank 1 \
  --device cuda:1 \
  --balance 12 12 11

python  test.py \
  --base_model /home/hzt/model/llama-7b-hf \
  --config ./config/dummy_pp.json \
  --pipeline \
  --rank 2 \
  --device cuda:2 \
  --balance 12 12 11

python test.py \
  --base_model /home/hzt/model/llama-7b-hf \
  --config ./config/dummy_pp.json \
  --pipeline \
  --rank 0 \
  --balance 17 18

python  test.py \
  --base_model /home/hzt/model/llama-7b-hf \
  --config ./config/dummy_pp.json \
  --pipeline \
  --rank 1 \
  --balance 17 18

python mlora.py \
  --base_model /home/hzt/model/llama-7b-hf \
  --config ./config/dummy.json

python .github/workflows/ci_script.py "llama" "/home/hzt/model/llama-7b-hf" "/home/hzt/workspace/test/multi-lora-fine-tune/lora_0/lora_0_final" "What is m-LoRA?"