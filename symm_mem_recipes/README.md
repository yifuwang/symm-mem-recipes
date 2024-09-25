## triton_all_gather_matmul.py

This is a fused all-gather matmul example using Triton + SymmetricMemory, based on the `tma_persistent` Triton tutorial with slight modifications.

This example requires PyTorch Nightly and Triton 3.0.0+ to run.

```bash
torchrun \
--nnodes 1 --nproc-per-node 8 \
--rdzv-backend c10d --rdzv-endpoint localhost:0 \
--no_python python3 -m symm_mem_recipes.triton_all_gather_matmul \
--M 16384 --N 6656 --K 16384 --BLOCK_SIZE_M 128 --BLOCK_SIZE_N 256 --BLOCK_SIZE_K 64
```

Some benchmarks on 8xH100 (special version with HBM2e, at 500W) with NVSwitch:

#### Llama 3 8B (N=1792, K=4096)
| Problem Size (M) | Block Size (M, N, K) | cublas mm only (us) | triton mm only (us) | cublas + nccl (us) | triton fused (us) |
|------------|------------|------------|------------|------------|------------|
| 4096 | 128, 128, 128 | 105 | 125 | 230 | 213 |
| 8192 | 128, 128, 128 | 194 | 236 | 416 | 318 |
| 16384 | 256, 128, 64 | 391 | 434 | 819 | 514 |

#### Llama 3 70B (N=3584, K=8192)
| Problem Size (M) | Block Size (M, N, K) | cublas mm only (us) | triton mm only (us) | cublas + nccl (us) | triton fused (us) |
|------------|------------|------------|------------|------------|------------|
| 4096 | 128, 128, 128 | 403 | 483 | 652 | 543 |
| 8192 | 256, 128, 64 | 828 | 849 | 1291 | 948 |
| 16384 | 256, 128, 64 | 1672 | 1655 | 2541 | 1846 |

#### Llama 3 105B (N=6656, K=16384)
| Problem Size (M) | Block Size (M, N, K) | cublas mm only (us) | triton mm only (us) | cublas + nccl (us) | triton fused (us) |
|------------|------------|------------|------------|------------|------------|
| 4096 | 128, 256, 64 | 1558 | 1595 | 2077 | 1776 |
| 8192 | 128, 256, 64 | 2879 | 2953 | 3847 | 3243 |
| 16384 | 128, 256, 64 | 5842 | 5948 | 7801 | 6538 |
