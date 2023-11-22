import gzip
import itertools
import json
import pathlib
from datetime import datetime

import pytz
import torch
from tqdm import tqdm

import punica.ops

from .benchmark_utils import bench, gc_torch, get_lora_lens


@torch.inference_mode()
def bench_sgmv(f):
    bs_ = list(range(1, 65))
    pop_ = ["bmm", "bgmv", "uniform", "zipf:1.5", "Nx8"]
    h1 = 4096
    h2 = 16
    num_layers = 1
    dtype = torch.float16
    device = torch.device("cuda:0")

    all_ = list(itertools.product(pop_, bs_))
    for pop, bs in (pbar := tqdm(all_)):
        if pop == "Nx8":
            if bs % 8 != 0:
                continue
            problem_sizes = [(bs // 8)] * 8
        else:
            problem_sizes = get_lora_lens(bs, pop)

        setup = dict(
            h1=h1,
            h2=h2,
            popularity=pop,
            num_problems=len(problem_sizes),
            batch_size=bs,
        )
        pbar.set_postfix(setup)

        torch.manual_seed(0xABCDABCD987)
        gc_torch()

        w = [
            torch.randn((num_layers, h2, h1), dtype=dtype, device=device)
            for _ in range(len(problem_sizes))
        ]
        w_ptr = torch.tensor(
            [t.data_ptr() for t in w], dtype=torch.int64, device=device
        )
        s = torch.cumsum(
            torch.tensor([0] + problem_sizes, device=device), dim=0, dtype=torch.int32
        )
        x = torch.randn((s[-1], h1), dtype=dtype, device=device)
        y = torch.randn((s[-1], h2), dtype=dtype, device=device)

        latency = bench(
            lambda: punica.ops.sgmv(y, x, w_ptr, s, layer_idx=0),
            warmup=200,
            repeat=1000,
        )

        result = {
            "setup": setup,
            "latency": {
                "avg": latency.avg(),
                "std": latency.std(),
            },
        }
        f.write(json.dumps(result) + "\n")
        f.flush()


def main():
    this_file = pathlib.Path(__file__)
    project_root = this_file.parents[1]
    now = datetime.now(pytz.timezone("US/Pacific"))
    out_filename = f"{now:%Y%m%d-%H%M%S}-{this_file.stem}.jsonl.gz"
    out_path = project_root / "data" / out_filename

    print(out_path)
    with gzip.open(out_path, "wt") as f:
        bench_sgmv(f)


if __name__ == "__main__":
    main()
