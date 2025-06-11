import functools

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem



def multimem_all_reduce(msg):
    torch.ops.symm_mem.multimem_all_reduce_(
        msg,
        "sum",
        dist.group.WORLD.group_name,
    )


def one_shot_all_reduce(msg):
    torch.ops.symm_mem.one_shot_all_reduce(
        msg,
        "sum",
        dist.group.WORLD.group_name,
    )


def two_shot_all_reduce(msg):
    torch.ops.symm_mem.two_shot_all_reduce_(
        msg,
        "sum",
        dist.group.WORLD.group_name,
    )


def triton_multimem_all_reduce(msg):
    from triton_multimem_all_reduce import multimem_all_reduce

    multimem_all_reduce(msg)


def triton_one_shot_all_reduce(msg):
    from triton_one_shot_all_reduce import one_shot_all_reduce

    one_shot_all_reduce(msg)


def nccl_ring(msg):
    functools.partial(dist.all_reduce, msg)


import os
from utils import benchmark_with_event
from collections import defaultdict
import argparse
import csv
from dataclasses import asdict, dataclass
from typing import Optional
from tabulate import tabulate

def formatt_large_number(num: int) -> str:
    if num >= 2**30:
        return f"{num / 2**30:.2f}g"
    if num >= 2**20:
        return f"{num / 2**20:.2f}m"
    if num >= 2**10:
        return f"{num / 2**10:.2f}k"
    return str(num)

@dataclass(frozen=True)
class ExperimentConfig:
    shape: tuple[int]
    dtype: torch.dtype
    backends: list[str]
    baseline_backend: str
    device: torch.device

    def asdict(self):
        # Convert the dataclass instance to a dictionary
        d = asdict(self)
        d.pop("backends", None)
        d.pop("device", None)
        d.pop("baseline_backend", None)

        formated_size = [formatt_large_number(num) for num in self.shape]
        d["shape"] = tuple(formated_size)
        return d


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    results: dict[str, float] # backend -> time in us

    def asdict(self):
        dict1 = self.config.asdict()
        dict2 = self.results
        return {**dict1, **dict2}


def generate_experiment_configs(
    dtype: torch.dtype,
    sizes: list[int],
    backends: list[str],
    device: torch.device
) -> list[ExperimentConfig]:

    all_configs = []
    for sz in sizes:
        all_configs.append(
            ExperimentConfig(
                shape=(sz,),
                dtype=dtype,
                backends=backends,
                baseline_backend=backends[0],
                device=device
            )
        )

    return all_configs


def get_single_backend_fn(backend: str):
    if backend == "dist_multimem":
        return multimem_all_reduce
    elif backend == "dist_1shot":
        return one_shot_all_reduce
    elif backend == "dist_2shot":
        return two_shot_all_reduce
    elif backend == "triton_multimem":
        return triton_multimem_all_reduce
    elif backend == "triton_1shot":
        return triton_one_shot_all_reduce
    elif backend == "nccl":
        return dist.all_reduce
    else:
        raise NotImplementedError(backend)


def run_experiment(
    config: ExperimentConfig
) -> dict[str, float]:
    input_tensor = symm_mem.empty(
        config.shape,
        dtype=config.dtype,
        device=config.device,
    )
    symm_mem.rendezvous(input_tensor, dist.group.WORLD.group_name)

    gloden_o = get_single_backend_fn(config.baseline_backend)(input_tensor)

    results = {}
    for backend in config.backends:
        fn = get_single_backend_fn(backend)
        target_fn = functools.partial(fn, input_tensor)

        test_o = target_fn()
        torch.testing.assert_close(test_o, gloden_o)

        results[backend] = benchmark_with_event(target_fn, flush_l2=True)

    return results


def print_results(results: list[Experiment], save_path: Optional[str] = None):
    table_data = defaultdict(list)

    for experiment in results:
        baseline_time = experiment.results[experiment.config.baseline_backend]
        min_time = float("inf")
        best_backend = experiment.config.baseline_backend
        backends = experiment.config.backends
        for key, value in experiment.asdict().items():
            if key in backends:
                if value < min_time:
                    min_time = value
                    best_backend = key
                table_data[key].append(value)
            else:
                table_data[key].append(value)
        table_data[f"Speedup over {experiment.config.baseline_backend}"].append(baseline_time / min_time)
        table_data["Best Backend"].append(best_backend)
    print(tabulate(table_data, headers="keys", tablefmt="github", floatfmt=".3f"))

    if save_path is not None:
        with open(save_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=table_data.keys())
            writer.writeheader()
            for i in range(len(next(iter(table_data.values())))):
                row = {k: v[i] for k, v in table_data.items()}
                writer.writerow(row)
        print(f"\nResults saved to {save_path}")


def main(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    torch.manual_seed(42 + local_rank)

    results = []
    configs = generate_experiment_configs(args.dtype, args.size, args.backend, device)
    for config in configs:
        results.append(
            Experiment(
                config,
                run_experiment(config),
            )
        )
    if dist.get_rank() == 0:
        print_results(results, args.save_path)


if __name__ == "__main__":
    """ Run with torchrun

    torchrun \
        --nnodes 1 --nproc-per-node 8 \
        --rdzv-backend c10d --rdzv-endpoint localhost:0 \
        --no_python python3 \
        benchmark_all_reduce.py --save-path allreduce.csv
    """


    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Run sweep over sizes for Allreduce. "
    )

    parser.add_argument(
        "--backend",
        type=str,
        nargs="+",
        choices=["nccl", "triton_1shot", "triton_multimem", "dist_multimem", "dist_1shot", "dist_2shot"],
        default=["nccl", "triton_multimem"],
        help="Backend to use for AllReduce. Use first backend as baseline. ",
    )

    parser.add_argument(
        "--size",
        type=int,
        nargs="+",
        default=[2**exp for exp in range(12, 21)],
        help="Tensor lengths"
    )

    parser.add_argument("-dtype", type=str, help="dtype", default="bfloat16")
    parser.add_argument(
        "--save-path",
        type=str,
        help="Path to save the results JSON file (optional)",
        default=None,
    )


    args = parser.parse_args()
    args.dtype = getattr(torch, args.dtype)

    main(args)
