#!/bin/bash

# Distributed tuning on multiple GPUs
# 

for nk in 24576,1536 2112,7168; do
    for m in 16, 32, 64, 128; do
        CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 python -u tune_triton_gemm_a8w8_blockscale.py -m 4 -nk 7168,16384 |tee 4.log &
    done
done


# CUDA_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=1 python -u tune_triton_gemm_a8w8_blockscale.py -m 8 |tee 8.log &
# CUDA_VISIBLE_DEVICES=2 HIP_VISIBLE_DEVICES=2 python -u tune_triton_gemm_a8w8_blockscale.py -m 16 |tee 16.log &
# CUDA_VISIBLE_DEVICES=3 HIP_VISIBLE_DEVICES=3 python -u tune_triton_gemm_a8w8_blockscale.py -m 64 |tee 64.log &
# CUDA_VISIBLE_DEVICES=4 HIP_VISIBLE_DEVICES=4 python -u tune_triton_gemm_a8w8_blockscale.py -m 32 |tee 32.log &

# Wait for all background jobs to complete
wait
echo "All tuning jobs completed!"