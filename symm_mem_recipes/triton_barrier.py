import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch._C._distributed_c10d import _SymmetricMemory
from torch.distributed._symmetric_memory import enable_symm_mem_for_group


@triton.jit
def get_flat_tid():
    return tl.inline_asm_elementwise(
        """
        {
            .reg .u32   %tmp32_<2>;

            mov.u32     %tmp32_0, %tid.z;
            mov.u32     %tmp32_1, %ntid.y;
            mul.lo.u32  %tmp32_0, %tmp32_0, %tmp32_1; // tid.z * ntid.y
            mov.u32     %tmp32_1, %ntid.x;
            mul.lo.u32  $0, %tmp32_0, %tmp32_1;       // $0 = tid.z * ntid.y * ntid.x
            mov.u32     %tmp32_0, %tid.y;
            mov.u32     %tmp32_1, %ntid.x;
            mul.lo.u32  %tmp32_0, %tmp32_0, %tmp32_1; // tid.y * ntid.x
            add.u32     $0, $0, %tmp32_0;             // $0 += tid.y * ntid.x
            mov.u32     %tmp32_0, %tid.x;
            add.u32     $0, $0, %tmp32_0;             // $0 += tid.x
        }
        """,
        "=r",
        [],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def blockwise_barrier(
    signal_pad_ptrs,
    block_id,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """
    A general purpose, efficient, CUDA graph-friendly multi-device barrier.

    CUDA graph friendliness:

        This barrier operates through atomic operations on a zero-filled signal
        pad, which resets to a zero-filled state after each successful
        synchronization. This design eliminates the need for incrementing a
        flag from host.

    Memory consistency:

        The barrier ensures a causality order between memory operations issued
        before the calling kernel across all devices and those issued after the
        barrier by all threads within the calling kernel. It achieves this
        through fine-grained acquire and release semantics, which is more
        efficient than __threadfence_system().

        To additionally ensure writes issued within the calling kernel across
        all devices are visible by all threads in the calling kernel after the
        barrier, a __threadfence_block() is required before the barrier.

    Psuedo code:

        if (warpid == 0 && laneid < world_size) {
            int remote_rank = laneid;
            cas &signal_pad_ptrs[remote_rank][block_id * world_size + rank] from 0 to 1 until succeed;
            cas &signal_pad_ptrs[rank][block_id * world_size + remote_rank] from 1 to 0 until succeed;
        }
        __syncthread()

        for (int remote_rank = 0; remote_rank < world_size; ++remote_rank) {
            acquire &signal_pad_ptrs[remote_rank][blockIdx.x * world_size + rank];
        }
    """
    if block_id is None:
        block_id = (
            tl.program_id(2) * tl.num_programs(1) * tl.num_programs(0)
            + tl.program_id(1) * tl.num_programs(0)
            + tl.program_id(0)
        )
    flat_tid = get_flat_tid()

    remote_ranks = tl.arange(0, WORLD_SIZE)
    signal_pad_ptrs = signal_pad_ptrs.to(tl.pointer_type(tl.uint64))
    remote_signal_pad_addrs = tl.load(signal_pad_ptrs + remote_ranks).to(
        tl.pointer_type(tl.uint32)
    )
    send_addrs = remote_signal_pad_addrs + block_id * WORLD_SIZE + RANK

    local_signal_pad_addr = tl.load(signal_pad_ptrs + RANK).to(
        tl.pointer_type(tl.uint32)
    )
    wait_addrs = local_signal_pad_addr + block_id * WORLD_SIZE + remote_ranks

    if flat_tid < WORLD_SIZE:
        tl.inline_asm_elementwise(
            """
            {
                .reg .u32   %tmp32_<1>;
                .reg .pred  %p<1>;

                send_signal:
                    atom.global.release.sys.cas.b32 %tmp32_0, [$1], 0, 1;
                    setp.eq.u32 %p0, %tmp32_0, 0;
                    @!%p0 bra send_signal;

                wait_signal:
                    // No need to acquire here since all threads will
                    // acquire this location after the barrier.
                    atom.global.sys.cas.b32 %tmp32_0, [$2], 1, 0;
                    setp.eq.u32 %p0, %tmp32_0, 1;
                    @!%p0 bra wait_signal;

                barrier_end:
            }
            """,
            "=r, l, l",
            [send_addrs, wait_addrs],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )

    tl.inline_asm_elementwise(
        "bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )

    for remote_rank in range(WORLD_SIZE):
        tl.inline_asm_elementwise(
            "ld.acquire.sys.global.u32 $0, [$1];",
            "=r, l",
            [local_signal_pad_addr + remote_rank],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )


@triton.jit
def barrier_test_kernel(
    signal_pad_ptrs,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    blockwise_barrier(signal_pad_ptrs, None, RANK, WORLD_SIZE)


def barrier_test(symm_mem: _SymmetricMemory):
    barrier_test_kernel[(32, 1, 1)](
        symm_mem.signal_pad_ptrs_dev,
        RANK=symm_mem.rank,
        WORLD_SIZE=symm_mem.world_size,
    )


if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    group_name = dist.group.WORLD.group_name
    enable_symm_mem_for_group(group_name)

    t = _SymmetricMemory.empty_strided_p2p(
        size=(4096,),
        stride=(1,),
        dtype=torch.uint32,
        device=device,
        group_name=group_name,
    ).fill_(0)

    symm_mem = _SymmetricMemory.rendezvous(t)
    barrier_test(symm_mem)
    print(t)
