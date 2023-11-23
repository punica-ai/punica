import gzip
import itertools
import json
import pathlib
from datetime import datetime

import pytz
import torch
from tqdm import tqdm

import punica.ops
from punica import BatchedKvCache, KvCache, KvPool

from .benchmark_utils import bench, gc_torch


class batch_decode_Resources:
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        page_len: int,
        seqlens: list[int],
        dtype: str,
        device: torch.device,
    ):
        dtype = getattr(torch, dtype)
        self.kvpool = KvPool(
            num_layers=1,
            num_heads=num_heads,
            head_dim=head_dim,
            page_len=page_len,
            dtype=dtype,
            device=device,
        )
        self.q = torch.randn(
            (len(seqlens), num_heads, head_dim), dtype=dtype, device=device
        )
        kv_list: list[KvCache] = []
        for seqlen in seqlens:
            kv_list.append(KvCache(self.kvpool, seqlen))
        self.kv_list = kv_list
        self.kv = BatchedKvCache(kv_list)

    def release(self):
        for kvcache in self.kv_list:
            kvcache.release()


@torch.inference_mode()
def bench_batch_decode(f):
    num_heads_ = [32, 40]
    batch_size_ = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        10,
        12,
        14,
        16,
        20,
        24,
        28,
        32,
        40,
        48,
        56,
        64,
    ]
    seqlen_ = list(reversed(range(2048, 0, -64)))
    dtype = "float16"
    device = torch.device("cuda:0")
    page_len = 16
    head_dim = 128

    all_ = list(itertools.product(num_heads_, seqlen_, batch_size_))
    for num_heads, seqlen, batch_size in (pbar := tqdm(all_)):
        setup = dict(
            num_heads=num_heads,
            head_dim=head_dim,
            page_len=page_len,
            seqlen=seqlen,
            batch_size=batch_size,
        )
        pbar.set_postfix(setup)
        torch.manual_seed(0xABCDABCD987)
        gc_torch()
        res = batch_decode_Resources(
            num_heads=num_heads,
            head_dim=head_dim,
            page_len=page_len,
            seqlens=[seqlen] * batch_size,
            dtype=dtype,
            device=device,
        )
        latency = bench(lambda: punica.ops.batch_decode(res.q, res.kv, layer_idx=0))
        res.release()

        result = {
            "setup": setup,
            "latency": {"avg": latency.avg(), "std": latency.std()},
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
        bench_batch_decode(f)


if __name__ == "__main__":
    main()
