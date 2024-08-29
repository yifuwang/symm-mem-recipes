import re
from contextlib import nullcontext
from typing import Callable, List, Optional

import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import get_cycles_per_ms


def benchmark_with_profiler(
    target_fn: Callable[[None], None],
    event_key_regex: str,
    benchmark_ranks: Optional[List[int]] = None,
    warmup_iters: int = 200,
    benchmark_iters: int = 25,
    flush_l2: bool = False,
) -> float:
    """
    Benchmark the target function with PyTorch profiler.

    Args:
        target_fn: The target function to benchmark.
        event_key_regex: The regex pattern to identify the profiler event
            associated with the target function.
        benchmark_ranks: The ranks to benchmark.
        warmup_iters: The number of warmup iterations.
        benchmark_iters: The number of benchmark iterations.
        flush_l2: Whether to flush the L2 cache before each invocation of the
            target function.

    Returns:
        The measured median latency in microseconds.
    """
    benchmark_ranks = benchmark_ranks or [0]

    if flush_l2:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")

    if dist.get_rank() in benchmark_ranks:
        try:
            from trace_handler import trace_handler
        except ImportError:
            trace_handler = None

        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=trace_handler,
        )
    else:
        prof = nullcontext()

    for _ in range(warmup_iters):
        target_fn()
    dist.barrier()
    torch.cuda.synchronize()

    with prof:
        torch.cuda._sleep(int(10 * get_cycles_per_ms()))
        for i in range(benchmark_iters):
            if flush_l2:
                cache.zero_()
            target_fn()
        torch.cuda.synchronize()

    if dist.get_rank() not in benchmark_ranks:
        return 0

    latencies_us = []
    for event in prof.events():
        if re.match(event_key_regex, event.key):
            latencies_us.append(event.device_time)

    if len(latencies_us) == 0:
        return 0

    return torch.tensor(latencies_us).median().item()
