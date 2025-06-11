import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

from triton_barrier import blockwise_barrier
from triton_utils import sync_threads
from utils import log_triton_kernel

@triton.jit
def one_shot_all_reduce_kernel(
    buffer_ptr_addrs,
    signal_pad_ptrs,
    output_ptr,
    numel: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
):
    blockwise_barrier(signal_pad_ptrs, None, rank, world_size, sem="relaxed")
    sync_threads()

    pid = tl.program_id(axis=0)
    buffer_ptr_addrs = buffer_ptr_addrs.to(tl.pointer_type(tl.uint64))
    output_ptr = output_ptr.to(tl.pointer_type(tl.bfloat16))
    block_start = pid * BLOCK_SIZE * NUMEL_PER_THREAD

    while block_start < numel:
        # Each thread processes 128 bits. Since Triton doesn't yet natively
        # support 128-bit dtypes, we achieve this by having each thread process
        # two 64-bit elements.

        # WHYY??
        # Each thread processes 128 bits -> 8 x bf16 elements.

        offsets = block_start + tl.arange(0, BLOCK_SIZE * NUMEL_PER_THREAD)
        mask = offsets < numel

        acc = tl.zeros((BLOCK_SIZE * NUMEL_PER_THREAD, ), dtype=tl.bfloat16)
        for i in range(world_size):

            buffer_ptr = tl.load(buffer_ptr_addrs + i).to(tl.pointer_type(tl.bfloat16))
            tl.multiple_of(buffer_ptr, 16)
            x = tl.load(buffer_ptr + offsets, mask=mask)
            acc += x
        tl.multiple_of(output_ptr, 16) # We're probably find without this.
        tl.store(output_ptr + offsets, acc, mask=mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE * NUMEL_PER_THREAD

    sync_threads()
    blockwise_barrier(signal_pad_ptrs, None, rank, world_size, sem="relaxed")


def one_shot_all_reduce(tensor: torch.Tensor):
    MAX_NUM_BLOCKS = 24
    NUM_WARPS = 32
    BLOCK_SIZE = NUM_WARPS * 32
    NUMEL_PER_THREAD = 8

    assert tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    assert (
        tensor.numel() % NUMEL_PER_THREAD == 0
    ), "The number of elements must be 128-bit aligned."
    num_blocks = min(
        triton.cdiv(triton.cdiv(tensor.numel(), NUMEL_PER_THREAD), BLOCK_SIZE),
        MAX_NUM_BLOCKS,
    )

    symm_mem_hdl = symm_mem.rendezvous(tensor, group=dist.group.WORLD)
    output = torch.empty_like(tensor)

    kernel = one_shot_all_reduce_kernel[(num_blocks, 1, 1)](
        symm_mem_hdl.buffer_ptrs_dev,
        symm_mem_hdl.signal_pad_ptrs_dev,
        output,
        numel=tensor.numel(),
        rank=symm_mem_hdl.rank,
        world_size=symm_mem_hdl.world_size,
        BLOCK_SIZE=BLOCK_SIZE,
        NUMEL_PER_THREAD=NUMEL_PER_THREAD,
        num_warps=NUM_WARPS,
    )
    log_triton_kernel(kernel)
    return output


if __name__ == "__main__":
    """
    torchrun \
    --nnodes 1 --nproc-per-node 8 \
    --rdzv-backend c10d --rdzv-endpoint localhost:0 \
    --no_python python3 triton_one_shot_all_reduce.py
    """
    from symm_mem_all_reduce import main

    main(["--impl", "triton_one_shot_all_reduce"])
