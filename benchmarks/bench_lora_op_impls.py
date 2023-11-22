import gzip
import itertools
import json
import pathlib
from datetime import datetime

import pytz
import torch
from tqdm import tqdm

import punica.ops

from .benchmark_utils import bench, get_lora_lens


def lora_loop(
    y: torch.Tensor,
    x: torch.Tensor,
    wa: torch.Tensor,
    wb: torch.Tensor,
    s: torch.IntTensor,
    layer_idx: int,
):
    for i in range(len(wa)):
        xi = x[s[i] : s[i + 1]]
        wai = wa[i][layer_idx, :, :]
        wbi = wb[i][layer_idx, :, :]
        y[s[i] : s[i + 1]] += xi @ wai @ wbi


def gather(
    wa_all: list[torch.Tensor],
    wb_all: list[torch.Tensor],
    problem_sizes: list[int],
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    wa, wb = [], []
    for i, l in enumerate(problem_sizes):
        wa.extend([wa_all[i][layer_idx, :, :]] * l)
        wb.extend([wb_all[i][layer_idx, :, :]] * l)
    wa = torch.stack(wa)
    wb = torch.stack(wb)
    return wa, wb


def bmm(
    y: torch.Tensor,
    x: torch.Tensor,
    wa: torch.Tensor,
    wb: torch.Tensor,
):
    y += (x.unsqueeze(1) @ wa @ wb).squeeze(1)


def lora_gbmm(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_all: list[torch.Tensor],
    wb_all: list[torch.Tensor],
    problem_sizes: list[int],
    layer_idx: int,
):
    wa, wb = gather(wa_all, wb_all, problem_sizes, layer_idx)
    bmm(y, x, wa, wb)


@torch.inference_mode()
def bench_lora_op_impls(f):
    bs_ = list(range(1, 65))
    pop_ = ["bmm", "bgmv", "uniform", "zipf:1.5", "Nx8"]
    h1 = 4096
    h2 = 11008
    r = 16
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
            r=r,
            popularity=pop,
            num_problems=len(problem_sizes),
            batch_size=bs,
        )
        pbar.set_postfix(setup)

        torch.manual_seed(0xABCDABCD987)
        wa = [
            torch.rand((num_layers, h1, r), dtype=dtype, device=device)
            for _ in range(len(problem_sizes))
        ]
        wb = [
            torch.rand((num_layers, r, h2), dtype=dtype, device=device)
            for _ in range(len(problem_sizes))
        ]
        wa_t = [t.transpose(-1, -2) for t in wa]
        wb_t = [t.transpose(-1, -2) for t in wb]
        wa_ptr = torch.tensor(
            [t.data_ptr() for t in wa_t], dtype=torch.int64, device=device
        )
        wb_ptr = torch.tensor(
            [t.data_ptr() for t in wb_t], dtype=torch.int64, device=device
        )
        s = torch.cumsum(
            torch.tensor([0] + problem_sizes, device=device), dim=0, dtype=torch.int32
        )
        x = torch.rand((s[-1], h1), dtype=dtype, device=device)
        y = torch.rand((s[-1], h2), dtype=dtype, device=device)
        gwa, gwb = gather(wa, wb, problem_sizes, layer_idx=0)

        l_loop = bench(
            lambda: lora_loop(y, x, wa, wb, s, layer_idx=0), warmup=200, repeat=1000
        )
        l_gbmm = bench(
            lambda: lora_gbmm(y, x, wa, wb, problem_sizes, layer_idx=0),
            warmup=200,
            repeat=1000,
        )
        l_gather = bench(
            lambda: gather(wa, wb, problem_sizes, layer_idx=0), warmup=200, repeat=1000
        )
        l_bmm = bench(lambda: bmm(y, x, gwa, gwb), warmup=200, repeat=1000)
        l_sgmv = bench(
            lambda: punica.ops.add_lora_sgmv(
                y, x, wa_ptr, wb_ptr, s, layer_idx=0, lora_rank=r
            ),
            warmup=200,
            repeat=1000,
        )

        result = {
            "setup": setup,
            "loop": dict(avg=l_loop.avg(), std=l_loop.std()),
            "gbmm": dict(avg=l_gbmm.avg(), std=l_gbmm.std()),
            "gather": dict(avg=l_gather.avg(), std=l_gather.std()),
            "bmm": dict(avg=l_bmm.avg(), std=l_bmm.std()),
            "sgmv": dict(avg=l_sgmv.avg(), std=l_sgmv.std()),
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
        bench_lora_op_impls(f)


if __name__ == "__main__":
    main()
