import gzip
import json
import pathlib
from datetime import datetime

import numpy as np
import pytz
import torch
from tqdm import tqdm

from .benchmark_utils import bench


@torch.inference_mode()
def bench_backbone_vs_lora(f):
    torch.manual_seed(0xABCDABCD987)
    dtype = torch.float16
    device = torch.device("cuda:0")
    h1 = 4096
    h2 = 11008
    r = 16
    bs_list = np.arange(1, 65)

    res = dict(
        backbone_avg=[],
        backbone_std=[],
        single_lora_avg=[],
        single_lora_std=[],
        multi_lora_avg=[],
        multi_lora_std=[],
    )
    for bs in tqdm(bs_list):
        w = torch.randn(h1, h2, dtype=dtype, device=device)
        wa = torch.randn(h1, r, dtype=dtype, device=device)
        wb = torch.randn(r, h2, dtype=dtype, device=device)
        x = torch.randn(bs, 1, h1, dtype=dtype, device=device)

        def muti_lora():
            for i in range(bs):
                x[i] @ wa @ wb

        l_backbone = bench(lambda: x @ w, warmup=200, repeat=500)
        l_single_lora = bench(lambda: x @ wa @ wb, warmup=200, repeat=500)
        l_multi_lora = bench(muti_lora, warmup=200, repeat=500)

        res["backbone_avg"].append(l_backbone.avg())
        res["backbone_std"].append(l_backbone.std())
        res["single_lora_avg"].append(l_single_lora.avg())
        res["single_lora_std"].append(l_single_lora.std())
        res["multi_lora_avg"].append(l_multi_lora.avg())
        res["multi_lora_std"].append(l_multi_lora.std())

    json.dump(res, f)


def main():
    this_file = pathlib.Path(__file__)
    project_root = this_file.parents[1]
    now = datetime.now(pytz.timezone("US/Pacific"))
    out_filename = f"{now:%Y%m%d-%H%M%S}-{this_file.stem}.json.gz"
    out_path = project_root / "data" / out_filename

    print(out_path)
    with gzip.open(out_path, "wt") as f:
        bench_backbone_vs_lora(f)


if __name__ == "__main__":
    main()
