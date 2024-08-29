import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch._C._distributed_c10d import _SymmetricMemory
from torch.distributed._symmetric_memory import (
    enable_symm_mem_for_group,
    get_symm_mem_workspace,
)

from .triton_barrier import blockwise_barrier
from .utils import benchmark_with_profiler


@triton.jit
def gemv_all_reduce_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    buffer_ptrs,
    signal_pad_ptrs,
    counter,
    N,
    K,  # Making N and K constexpr seems to result in slower kernel
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N_RED: tl.constexpr,
):
    pid = tl.program_id(0)
    npids = tl.num_programs(0)

    n_start = pid * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)[:, None]
    n_mask = n_offsets < N
    acc = tl.zeros([BLOCK_N, BLOCK_K], tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)[None, :]
        k_mask = k_offsets < K
        a = tl.load(
            x_ptr + (k_offsets), k_mask, eviction_policy="evict_last", other=0.0
        ).to(tl.float32)
        b = tl.load(
            w_ptr + (k_offsets + (K * n_offsets)),
            k_mask & n_mask,
            eviction_policy="evict_first",
            other=0.0,
        ).to(tl.float32)
        c = a * b
        acc_ = acc + tl.broadcast_to(c, [BLOCK_N, BLOCK_K])
        acc = tl.where(k_mask & n_mask, acc_, acc)

    acc = tl.sum(acc, 1)[:, None]

    # Broadcast the acc to all rank's workspace
    buffer_ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))
    buffer_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(tl.bfloat16))
    remote_buffer_ptrs = tl.load(buffer_ptrs + tl.arange(0, world_size)).to(
        tl.pointer_type(tl.bfloat16)
    )
    tl.store(
        n_offsets + rank * N + remote_buffer_ptrs[None, :],
        acc.broadcast_to([BLOCK_N, world_size]),
        n_mask,
    )

    # Report progress to the reduction blocks. Note that Triton already inserts
    # a bar.sync after the store. It establishes observation order between the
    # writes by this cta and the atomic add. Acquiring the atomic add would
    # make the writes visible.
    tl.atomic_add(counter, 1, sem="release")

    # The last N // BLOCK_N_RED blocks performs reduction
    if pid >= npids - N // BLOCK_N_RED:
        red_block_id = npids - pid - 1

        val = tl.atomic_add(counter, 0)
        while val < npids:
            val = tl.atomic_add(counter, 0)
        blockwise_barrier(signal_pad_ptrs, red_block_id, rank, world_size)

        n_start_ = red_block_id * BLOCK_N_RED
        n_offsets_ = n_start_ + tl.arange(0, BLOCK_N_RED)
        w_offsets = tl.arange(0, world_size)
        block_offsets = w_offsets[:, None] * N + n_offsets_[None, :]
        block = tl.load(buffer_ptr + block_offsets, n_offsets_[None, :] < N)
        vec = tl.sum(block, 0)
        tl.store(out_ptr + n_start_ + tl.arange(0, BLOCK_N_RED), vec, n_offsets_ < N)


ptx = None


def gemv_all_reduce(
    x: torch.Tensor, w: torch.Tensor, output: torch.Tensor, workspace: _SymmetricMemory
):
    N, K = w.shape[0], w.shape[1]
    counter = torch.zeros(1, dtype=torch.uint32, device=x.device)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    compiled = gemv_all_reduce_kernel[grid](
        x,
        w,
        output,
        workspace.buffer_ptrs_dev,
        workspace.signal_pad_ptrs_dev,
        counter,
        N,
        K,
        rank=workspace.rank,
        world_size=workspace.world_size,
        BLOCK_N=2,
        BLOCK_K=2048,
        BLOCK_N_RED=2048,
        num_warps=16,
        num_ctas=1,
    )
    global ptx
    ptx = compiled.asm["ptx"]
    return output


if __name__ == "__main__":
    """
    torchrun \
    --nnodes 1 --nproc-per-node 8 \
    --rdzv-backend c10d --rdzv-endpoint localhost:0 \
    --no_python python3 -m symm_mem_recipes.triton_gemv_all_reduce
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    group_name = dist.group.WORLD.group_name
    enable_symm_mem_for_group(group_name)
    torch.manual_seed(42 + rank)

    M, N, K = 1, 16384, 8192

    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    output = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    workspace = get_symm_mem_workspace(group_name, world_size * N * 2)

    ########
    # def fn():
    #     output = torch.mm(x, w.t())
    #     dist.all_reduce(output)
    #     return output

    # print(fn())
    # benchmark_with_profiler(lambda: fn(), "", benchmark_iters=200)
    # import sys
    # sys.exit(0)
    ########

    ########
    # import torch._inductor.config as config
    # config.coordinate_descent_tuning = True

    # def f(x, w):
    #     return torch.mm(x, w.t())

    # cf = torch.compile(f)
    # lat_us = benchmark_with_profiler(lambda: cf(x, w), "triton", benchmark_iters=200)
    # if rank == 0:
    #     print(f"Median latency: {lat_us:.2f} us")
    # import sys
    # sys.exit(0)
    ########

    gemv_all_reduce(x, w, output, workspace)
    print(output)

    lat_us = benchmark_with_profiler(
        lambda: gemv_all_reduce(x, w, output, workspace),
        "gemv_all_reduce_kernel",
        benchmark_iters=200,
    )
    if rank == 0:
        print(f"Median latency: {lat_us:.2f} us")

    dist.destroy_process_group()
