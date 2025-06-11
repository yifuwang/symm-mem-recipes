import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

from triton_utils import get_flat_bid, get_flat_tid, sync_threads
from utils import log_triton_kernel


@triton.jit
def wait_gmem_barrier(addr):
    # tl.device_print("consumer load", tl.load(addr))

    while tl.atomic_add(addr, 0, sem="acquire", scope="gpu") == 0:
        pass # -> ld.global.acquire ...
    tl.debug_barrier()

@triton.jit
def signal_gmem_barrier(addr):
    tl.debug_barrier()
    tl.atomic_xchg(addr, 1, sem="release", scope="gpu")

@triton.jit
def local_barrier_test_kernel(
    signal_pad_ptrs,
    NUM_PRODUCERS: tl.constexpr,
    NUM_CONSUMERS: tl.constexpr,
):
    bid = tl.program_id(0)
    if (bid < NUM_PRODUCERS):
        signal_gmem_barrier(signal_pad_ptrs + bid )
        pass
    else:
        consumer_id = bid - NUM_PRODUCERS
        producer_id = consumer_id // (NUM_CONSUMERS // NUM_PRODUCERS)
        wait_gmem_barrier(signal_pad_ptrs + producer_id )



def local_barrier_test() -> None:
    signal_pad = torch.zeros(8, device="cuda", dtype=torch.int32).contiguous()
    num_producers = 8
    num_consumers = 32




    kernel = local_barrier_test_kernel[(num_producers+num_consumers, 1, 1)](
        signal_pad,
        NUM_PRODUCERS = num_producers,
        NUM_CONSUMERS = num_consumers,

    )
    log_triton_kernel(kernel)

    print(signal_pad)



if __name__ == "__main__":
    local_barrier_test()
