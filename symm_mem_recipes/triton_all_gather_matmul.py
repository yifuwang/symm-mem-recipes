import os

import click
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton.tools.experimental_descriptor
from torch._C._distributed_c10d import _SymmetricMemory
from torch.distributed._symmetric_memory import (
    _get_backend_stream,
    enable_symm_mem_for_group,
)

from symm_mem_recipes.utils import benchmark_with_event

from .triton_barrier import get_flat_tid


def all_gather_with_progress(
    output: torch.Tensor,
    inp: torch.Tensor,
    progress: torch.Tensor,
    splits_per_rank: int,
):
    assert inp.is_contiguous()

    symm_mem = _SymmetricMemory.rendezvous(inp)
    assert symm_mem is not None

    rank = symm_mem.rank
    world_size = symm_mem.world_size

    assert inp.numel() % splits_per_rank == 0
    assert progress.numel() == world_size * splits_per_rank

    output_shape = list(inp.shape)
    output_shape[0] *= world_size
    assert list(output.shape) == output_shape, (list(output.shape), output_shape)

    chunks = output.chunk(world_size * splits_per_rank)

    for step in range(0, world_size):
        src_rank = (rank + step + 1) % world_size
        for split_id in range(splits_per_rank):
            src_buf = symm_mem.get_buffer(
                src_rank, chunks[0].shape, inp.dtype, chunks[0].numel() * split_id
            )
            chunks[src_rank * splits_per_rank + split_id].copy_(src_buf)
            # cuStreamWriteValue32 issues a system level fence before the write
            symm_mem.stream_write_value32(
                int(progress.data_ptr())
                + (src_rank * splits_per_rank + split_id) * progress.element_size(),
                1,
            )
    symm_mem.barrier()


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    ret["flops8"] = 2.0 * M * N * K
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret["bytes"] = bytes_per_elem * (M * K + N * K)
    return ret


@triton.jit
def wait_signal(addr, flat_tid):
    if flat_tid == 0:
        tl.inline_asm_elementwise(
            """
            {
                .reg .pred  %p<1>;

                wait_block:
                    ld.global.relaxed.gpu.u32 $0, [$1];
                    setp.eq.u32 %p0, $0, 1;
                    @!%p0 bra wait_block;
            }
            """,
            "=r, l",
            [addr],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )

    tl.inline_asm_elementwise(
        "bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_tma_persistent(
    a_shard_desc_ptr,
    a_desc_ptr,
    b_desc_ptr,
    c_desc_ptr,
    progress_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_BLOCK_SIZE_M: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """
    Slightly modified from the sm90 tma persistent Triton tutorial.
    """
    flat_tid = get_flat_tid()

    dtype = tl.float8e4nv if FP8_OUTPUT else tl.bfloat16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am_src = 0
    offs_bn = 0
    a_ptr = a_desc_ptr

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            NUM_COMM_BLOCKS = M // COMM_BLOCK_SIZE_M
            NUM_COMM_BLOCKS_PER_RANK = NUM_COMM_BLOCKS // WORLD_SIZE
            NUM_PID_M_PER_COMM_BLOCK = COMM_BLOCK_SIZE_M // BLOCK_SIZE_M

            # Pivot tile_id so that M tiles are processed in their ready order.
            # This pivot preserves the prior swizzling.
            pid_m = (pid_m + NUM_PID_M_PER_COMM_BLOCK * RANK) % num_pid_m

            comm_block_id = pid_m // NUM_PID_M_PER_COMM_BLOCK
            if comm_block_id // NUM_COMM_BLOCKS_PER_RANK == RANK:
                # Read from the local a_shard
                offs_am_src = (pid_m * BLOCK_SIZE_M) % COMM_BLOCK_SIZE_M
                a_ptr = a_shard_desc_ptr
            else:
                # Wait for and read from a_shard copied from remote ranks
                wait_signal((progress_ptr + comm_block_id).to(tl.uint64), flat_tid)
                offs_am_src = pid_m * BLOCK_SIZE_M
                a_ptr = a_desc_ptr

        offs_bn = pid_n * BLOCK_SIZE_N
        offs_k = ki * BLOCK_SIZE_K

        a = tl._experimental_descriptor_load(
            a_ptr, [offs_am_src, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], dtype
        )
        b = tl._experimental_descriptor_load(
            b_desc_ptr, [offs_bn, offs_k], [BLOCK_SIZE_N, BLOCK_SIZE_K], dtype
        )
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            c = accumulator.to(dtype)

            tl._experimental_descriptor_store(
                c_desc_ptr, c, [pid_m * BLOCK_SIZE_M, offs_bn]
            )
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


_tma_desc_cache = {}


def create_2d_tma_descriptor(ptr, dim1, dim0, block_dim1, block_dim0, element_size):
    global _tma_desc_cache
    key = (ptr, dim1, dim0, block_dim1, block_dim0, element_size)
    if key in _tma_desc_cache:
        return _tma_desc_cache[key]
    desc = triton.tools.experimental_descriptor.create_2d_tma_descriptor(
        ptr,
        dim1,
        dim0,
        block_dim1,
        block_dim0,
        element_size,
    )
    _tma_desc_cache[key] = desc
    return desc


last_ptx = None


def all_gather_matmul_tma_persistent(
    a_shard, b, a_out, c_out, configs, mm_only: bool = False
):
    if mm_only:
        rank = 0
        world_size = int(os.environ.get("WORLD_SIZE", "8"))
    else:
        symm_mem = _SymmetricMemory.rendezvous(a_shard)
        assert symm_mem is not None, "a_shard must be allocated via SymmetricMemory"
        rank = symm_mem.rank
        world_size = symm_mem.world_size

    dtype = a_shard.dtype
    M = a_shard.shape[0] * world_size
    N = b.shape[0]
    K = a_shard.shape[1]

    assert b.shape[1] == K
    assert a_out.shape[0] == M
    assert a_out.shape[1] == K
    assert c_out.shape[0] == M
    assert c_out.shape[1] == N

    SPLITS_PER_RANK = 1
    COMM_BLOCK_SIZE_M = M // world_size // SPLITS_PER_RANK
    assert COMM_BLOCK_SIZE_M % (configs["BLOCK_SIZE_M"] * configs["GROUP_SIZE_M"]) == 0

    if mm_only:
        progress = torch.ones(world_size, dtype=torch.uint32, device="cuda")
    else:
        progress = torch.zeros(world_size, dtype=torch.uint32, device="cuda")
        symm_mem.barrier(0)
        _get_backend_stream().wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(_get_backend_stream()):
            all_gather_with_progress(a_out, a_shard, progress, SPLITS_PER_RANK)

    desc_a_shard = create_2d_tma_descriptor(
        a_shard.data_ptr(),
        a_shard.shape[0],
        K,
        configs["BLOCK_SIZE_M"],
        configs["BLOCK_SIZE_K"],
        a_shard.element_size(),
    )
    desc_a = create_2d_tma_descriptor(
        a_out.data_ptr(),
        M,
        K,
        configs["BLOCK_SIZE_M"],
        configs["BLOCK_SIZE_K"],
        a_out.element_size(),
    )
    desc_b = create_2d_tma_descriptor(
        b.data_ptr(),
        N,
        K,
        configs["BLOCK_SIZE_N"],
        configs["BLOCK_SIZE_K"],
        b.element_size(),
    )
    desc_c = create_2d_tma_descriptor(
        c_out.data_ptr(),
        M,
        N,
        configs["BLOCK_SIZE_M"],
        configs["BLOCK_SIZE_N"],
        c_out.element_size(),
    )
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    grid = lambda META: (
        min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ),
    )
    compiled = matmul_kernel_tma_persistent[grid](
        desc_a_shard,
        desc_a,
        desc_b,
        desc_c,
        progress,
        M,
        N,
        K,
        BLOCK_SIZE_M=configs["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=configs["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=configs["BLOCK_SIZE_K"],
        GROUP_SIZE_M=configs["GROUP_SIZE_M"],
        COMM_BLOCK_SIZE_M=COMM_BLOCK_SIZE_M,
        RANK=rank,
        WORLD_SIZE=world_size,
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,
        NUM_SMS=NUM_SMS,
        num_stages=configs["num_stages"],
        num_warps=configs["num_warps"],
    )
    global last_ptx
    last_ptx = compiled.asm["ptx"]
    torch.cuda.current_stream().wait_stream(_get_backend_stream())
    return c_out


def all_gather_matmul(a_shard, b):
    from torch.distributed._functional_collectives import all_gather_tensor

    a = all_gather_tensor(a_shard, 0, "0")
    return torch.matmul(a, b)


@click.command()
@click.option("--M", default=4096)
@click.option("--N", default=6656)
@click.option("--K", default=16384)
@click.option("--BLOCK_SIZE_M", default=128)
@click.option("--BLOCK_SIZE_N", default=256)
@click.option("--BLOCK_SIZE_K", default=64)
@click.option("--GROUP_SIZE_M", default=4)
@click.option("--num_stages", default=3)
@click.option("--num_warps", default=8)
@click.option("--print_ptx", is_flag=True)
def main(
    m: int,
    n: int,
    k: int,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    group_size_m: int,
    num_stages: int,
    num_warps: int,
    print_ptx: bool,
):
    """
    torchrun \
    --nnodes 1 --nproc-per-node 8 \
    --rdzv-backend c10d --rdzv-endpoint localhost:0 \
    --no_python python3 -m symm_mem_recipes.triton_all_gather_matmul
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(42 + rank)
    dist.init_process_group("nccl")
    group_name = dist.group.WORLD.group_name
    enable_symm_mem_for_group(group_name)

    a_shard = _SymmetricMemory.empty_strided_p2p(
        size=(m // world_size, k),
        stride=(k, 1),
        dtype=torch.bfloat16,
        device=device,
        group_name=group_name,
    ).normal_()
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16).T.contiguous()
    c = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)

    # Autotuner does not work with TMA. Use manual config.
    configs = {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": block_size_n,
        "BLOCK_SIZE_K": block_size_k,
        "GROUP_SIZE_M": group_size_m,
        "num_stages": num_stages,
        "num_warps": num_warps,
    }

    c0 = all_gather_matmul(a_shard, b.T)
    c1 = all_gather_matmul_tma_persistent(a_shard, b, a, c, configs)
    assert torch.allclose(c0, c1)

    lat_us = benchmark_with_event(lambda: torch.matmul(a, b.T, out=c), flush_l2=True)
    if rank == 0:
        print(f"cublas matmul only: {lat_us} us")

    lat_us = benchmark_with_event(
        lambda: all_gather_matmul_tma_persistent(
            a_shard, b, a, c, configs, mm_only=True
        ),
        flush_l2=True,
    )
    if rank == 0:
        print(f"triton matmul only: {lat_us} us")

    lat_us = benchmark_with_event(
        lambda: all_gather_matmul(a_shard, b.T), flush_l2=True
    )
    if rank == 0:
        print(f"cublas + nccl: {lat_us} us")

    lat_us = benchmark_with_event(
        lambda: all_gather_matmul_tma_persistent(a_shard, b, a, c, configs),
        flush_l2=True,
    )
    if rank == 0:
        print(f"triton all_gather_matmul: {lat_us} us")

    if print_ptx and rank == 0:
        print(last_ptx)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
