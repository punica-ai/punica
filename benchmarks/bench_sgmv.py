import gzip
import itertools
import json
import pathlib
from datetime import datetime

import pytz
import torch
from tqdm import tqdm

import punica.ops

from .benchmark_utils import bench, gc_torch


@torch.inference_mode()
def bench_sgmv(f):
  var_ = [
      1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64
  ]
  pop_ = ["bmm", "bgmv", "uniform"]
  h1 = 16
  h2 = 4096
  num_layers = 1
  dtype = torch.float16
  device = torch.device("cuda:0")

  all_ = list(itertools.product(pop_, var_))
  for pop, var in (pbar := tqdm(all_)):
    if pop == "bmm":
      num_problems = 1
      problem_size = var
    elif pop == "bgmv":
      num_problems = var
      problem_size = 1
    elif pop == "uniform":
      num_problems = var
      problem_size = var
    batch_size = num_problems * problem_size

    setup = dict(
        h1=h1,
        h2=h2,
        pop=pop,
        num_problems=num_problems,
        problem_size=problem_size,
        batch_size=batch_size,
    )
    pbar.set_postfix(setup)

    torch.manual_seed(0xabcdabcd987)
    gc_torch()

    w = [
        torch.randn((num_layers, h1, h2), dtype=dtype, device=device)
        for _ in range(num_problems)
    ]
    w_ptr = torch.tensor([t.data_ptr() for t in w],
                         dtype=torch.int64,
                         device=device)
    s = torch.cumsum(
        torch.tensor([0] + [problem_size] * num_problems, device=device),
        dim=0,
        dtype=torch.int32)
    x = torch.randn((s[-1], h1), dtype=dtype, device=device)
    y = torch.randn((s[-1], h2), dtype=dtype, device=device)

    latency = bench(
        lambda: punica.ops.sgmv_cutlass(y, x, w_ptr, s, layer_idx=0),
        warmup=1000,
        repeat=5000)

    result = {"setup": setup, "avg": latency.avg(), "std": latency.std()}
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
