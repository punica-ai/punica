import gzip
import itertools
import json
import pathlib
from datetime import datetime

import pytz
import torch
from tqdm import tqdm
from transformers import LlamaConfig

from punica import (
    BatchedKvCache,
    BatchedLlamaLoraWeight,
    BatchLenInfo,
    KvCache,
    KvPool,
    LlamaLoraWeight,
)
from punica.models.llama_lora import LlamaDecoderLayerWithLora

from .benchmark_utils import bench, gc_torch, get_lora_lens


class layer_lora_decode_Resources:
    def __init__(
        self,
        config: LlamaConfig,
        page_len: int,
        lora_rank: int,
        lora_popularity: int,
        seqlens: list[int],
        dtype: torch.dtype,
        device: torch.device,
    ):
        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        self.kvpool = KvPool(
            num_layers=1,
            num_heads=num_heads,
            head_dim=head_dim,
            page_len=page_len,
            dtype=dtype,
            device=device,
        )
        bs = len(seqlens)
        self.hidden_states = torch.randn(
            (bs, num_heads * head_dim), dtype=dtype, device=device
        )
        kv_list: list[KvCache] = []
        for seqlen in seqlens:
            kv_list.append(KvCache(self.kvpool, seqlen))
        self.kv_list = kv_list
        self.kv = BatchedKvCache(kv_list)
        self.blen = BatchLenInfo([], bs, device)

        lora_lens = get_lora_lens(bs, lora_popularity)
        self.num_lora_models = len(lora_lens)
        weights = [
            LlamaLoraWeight(config, lora_rank, dtype, device)
            for _ in range(self.num_lora_models)
        ]
        self.lora = BatchedLlamaLoraWeight(weights, lora_lens)

    def release(self):
        for kvcache in self.kv_list:
            kvcache.release()


@torch.inference_mode()
def bench_layer_lora_decode(f):
    num_heads_ = [32, 40]
    intermediate_size_ = [11008, 13824]
    pop_ = ["bgmv", "bmm", "uniform", "zipf:1.5"]
    batch_size_ = list(range(1, 64))
    seqlen_ = list(reversed(range(2048, 0, -64)))
    dtype = torch.float16
    device = torch.device("cuda:0")
    page_len = 16
    lora_rank = 16
    head_dim = 128

    all_ = list(
        itertools.product(
            zip(num_heads_, intermediate_size_), pop_, seqlen_, batch_size_
        )
    )
    last_num_heads = 0
    for (num_heads, intermediate_size), pop, seqlen, batch_size in (pbar := tqdm(all_)):
        if last_num_heads != num_heads:
            config = LlamaConfig(
                hidden_size=num_heads * head_dim,
                num_attention_heads=num_heads,
                intermediate_size=intermediate_size,
                num_hidden_layers=1,
            )
            default_dtype = torch.get_default_dtype()
            torch.set_default_dtype(dtype)
            with device:
                model = LlamaDecoderLayerWithLora(config, layer_idx=0).to(device)
            torch.set_default_dtype(default_dtype)

        torch.manual_seed(0xABCDABCD987)
        gc_torch()
        res = layer_lora_decode_Resources(
            config=config,
            page_len=page_len,
            lora_rank=lora_rank,
            lora_popularity=pop,
            seqlens=[seqlen] * batch_size,
            dtype=dtype,
            device=device,
        )
        setup = dict(
            num_heads=num_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            page_len=page_len,
            lora_rank=lora_rank,
            lora_popularity=pop,
            num_lora_models=res.num_lora_models,
            seqlen=seqlen,
            batch_size=batch_size,
        )
        pbar.set_postfix(setup)

        latency = bench(
            lambda: model(res.hidden_states, res.blen, None, res.kv, res.lora),
            warmup=10,
            repeat=50,
        )
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
        bench_layer_lora_decode(f)


if __name__ == "__main__":
    main()
