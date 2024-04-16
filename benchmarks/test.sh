CUDA_VISIBLE_DEVICES=0,1,3,4 torchrun --nnodes 1 --nproc_per_node 4 ./benchmarks/bench_peft_fsdp.py
CUDA_VISIBLE_DEVICES=0,1,3,4 torchrun --nnodes 1 --nproc_per_node 4 ./benchmarks/bench_mlora_pp.py
