# ruff: noqa: F821
import gzip
import itertools
import json
import pathlib
from datetime import datetime

import pytz
import torch
from tqdm import tqdm
from transformers import LlamaConfig

from punica import BatchedKvCache, BatchLenInfo, KvCache, KvPool

from .benchmark_utils import bench, gc_torch


class model_Resources:
    def __init__(
        self,
        config: LlamaConfig,
        page_len: int,
        seqlen: int,
        prefills: int,
        decodes: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        self.kvpool = KvPool(
            num_layers=config.num_hidden_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            page_len=page_len,
            dtype=dtype,
            device=device,
        )
        self.input_ids = torch.randint(
            0, 32000, (seqlen * prefills + decodes,), dtype=torch.int64, device=device
        )
        self.prefill_kv = (
            BatchedKvCache([KvCache(self.kvpool, seqlen) for _ in range(prefills)])
            if prefills
            else None
        )
        self.decode_kv = (
            BatchedKvCache([KvCache(self.kvpool, seqlen) for _ in range(decodes)])
            if decodes
            else None
        )
        self.blen = BatchLenInfo([seqlen] * prefills, decodes, device)


@torch.inference_mode()
def bench_model_prefill_decode(f):
    num_heads_ = [32]
    num_layers_ = [32]
    intermediate_size_ = [11008]
    prefill_decode_ = [(0, 1), (1, 0)]
    batch_size_ = list(range(1, 33))
    seqlen_ = [128, 512, 1024, 1536, 2048]
    dtype = torch.float16
    device = torch.device("cuda:0")
    page_len = 16
    head_dim = 128

    all_ = list(
        itertools.product(
            zip(num_heads_, num_layers_, intermediate_size_),
            prefill_decode_,
            seqlen_,
            batch_size_,
        )
    )
    last_num_heads = 0
    model = None
    for (num_heads, num_layers, intermediate_size), (
        prefill,
        decode,
    ), seqlen, batch_size in (pbar := tqdm(all_)):
        if last_num_heads != num_heads:
            config = LlamaConfig(
                hidden_size=num_heads * head_dim,
                num_attention_heads=num_heads,
                intermediate_size=intermediate_size,
                num_hidden_layers=num_layers,
            )
            del model
            gc_torch()
            default_dtype = torch.get_default_dtype()
            torch.set_default_dtype(dtype)
            with device:
                model = LlamaForCausalLM(config).to(device)
            torch.set_default_dtype(default_dtype)

        torch.manual_seed(0xABCDABCD987)
        gc_torch()
        res = model_Resources(
            config=config,
            page_len=page_len,
            seqlen=seqlen,
            prefills=batch_size * prefill,
            decodes=batch_size * decode,
            dtype=dtype,
            device=device,
        )
        setup = dict(
            num_heads=num_heads,
            head_dim=head_dim,
            num_layers=num_layers,
            intermediate_size=intermediate_size,
            page_len=page_len,
            seqlen=seqlen,
            prefills=batch_size * prefill,
            decodes=batch_size * decode,
            batch_size=batch_size,
        )
        pbar.set_postfix(setup)

        latency = bench(
            lambda: model(res.input_ids, res.blen, res.prefill_kv, res.decode_kv),
            warmup=1,
            repeat=5,
        )
        del res

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
        bench_model_prefill_decode(f)


if __name__ == "__main__":
    main()
