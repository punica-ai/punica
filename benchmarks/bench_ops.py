import argparse
import itertools

import torch

import punica.ops
import punica.ops._kernels

from .benchmark_utils import bench, gc_torch


class add_lora_Resources:

  def __init__(
      self,
      num_layers: int,
      in_features: int,
      out_features: int,
      lora_rank: int,
      batch_size: int,
      dtype: str,
      device: torch.device,
  ):
    dtype = getattr(torch, dtype)
    self.x = torch.randn((batch_size, in_features), dtype=dtype, device=device)
    self.y = torch.randn((batch_size, out_features), dtype=dtype, device=device)
    self.wa_T_all = torch.randn(
        (batch_size, num_layers, lora_rank, in_features),
        dtype=dtype,
        device=device)
    self.wb_T_all = torch.randn(
        (batch_size, num_layers, out_features, lora_rank),
        dtype=dtype,
        device=device)
    self.indicies = torch.arange(0, batch_size, dtype=torch.long, device=device)


def bench_add_lora():
  lora_ranks = [16]
  weight_sizes = [
      (4096, 4096),
      (4096, 11008),
      (11008, 4096),
  ]
  batch_sizes = list(range(1, 17))
  device = torch.device("cuda:0")

  print("bench_add_lora")
  for lora_rank, (in_features, out_features), batch_size in itertools.product(
      lora_ranks, weight_sizes, batch_sizes):
    torch.manual_seed(0xabcdabcd987)
    outputs = [
        f"r={lora_rank}",
        f"h1={in_features}",
        f"h2={out_features}",
        f"bs={batch_size:2d}",
    ]
    try:
      gc_torch()
      t = add_lora_Resources(
          num_layers=1,
          in_features=in_features,
          out_features=out_features,
          lora_rank=lora_rank,
          batch_size=batch_size,
          dtype="float16",
          device=device,
      )
      result = bench(lambda: punica.ops.add_lora(
          t.y, t.x, t.wa_T_all, t.wb_T_all, t.indicies, layer_idx=0, scale=1.0))
      outputs.append(f"{result.avg()*1e6:3.0f}us±{result.std()*1e6:3.0f}us")
    except torch.cuda.OutOfMemoryError:
      outputs.append("OOM")

    print(" | ".join(outputs))


class bgmv_Resources:

  def __init__(
      self,
      num_layers: int,
      in_features: int,
      out_features: int,
      batch_size: int,
      dtype: str,
      device: torch.device,
  ):
    dtype = getattr(torch, dtype)
    self.x = torch.randn((batch_size, in_features), dtype=dtype, device=device)
    self.y = torch.randn((batch_size, out_features), dtype=dtype, device=device)
    self.w_T_all = torch.randn(
        (batch_size, num_layers, out_features, in_features),
        dtype=dtype,
        device=device)
    self.indicies = torch.arange(0, batch_size, dtype=torch.long, device=device)


def bench_bgmv():
  lora_ranks = [8, 16, 32, 64]
  weight_sizes = [
      768,
      1024,
      2048,
      2560,
      3072,
      4096,
      5120,
      7168,
      8192,
      9216,
      10240,
      11008,
      12288,
      13824,
      16384,
      20480,
      28672,
      36864,
      49152,
  ]
  batch_sizes = list(range(1, 17))
  device = torch.device("cuda:0")

  print("bench_bgmv")
  all_sizes = list(
      itertools.chain(
          itertools.product(lora_ranks, weight_sizes),
          itertools.product(weight_sizes, lora_ranks)))
  for (in_features,
       out_features), batch_size in itertools.product(all_sizes, batch_sizes):
    torch.manual_seed(0xabcdabcd987)
    outputs = [
        f"h1={in_features}",
        f"h2={out_features}",
        f"bs={batch_size:2d}",
    ]
    try:
      gc_torch()
      t = bgmv_Resources(
          num_layers=1,
          in_features=in_features,
          out_features=out_features,
          batch_size=batch_size,
          dtype="float16",
          device=device,
      )
      layer_idx = 0
      scale = 1.0
      result = bench(lambda: punica.ops._kernels.dispatch_bgmv(
          t.y, t.x, t.w_T_all, t.indicies, layer_idx, scale))
      outputs.append(f"{result.avg()*1e6:3.0f}us±{result.std()*1e6:3.0f}us")
    except torch.cuda.OutOfMemoryError:
      outputs.append("OOM")

    print(" | ".join(outputs))


BENCH_FN = {
    "mha": bench_rotary_mha_decode_fixed_length,
    "lora": bench_add_lora,
    "bgmv": bench_bgmv,
}


def bench_one_op():
  parser = argparse.ArgumentParser()
  parser.add_argument("--op", choices=BENCH_FN.keys(), required=True)
  args = parser.parse_args()
  bench_fn = BENCH_FN[args.op]
  bench_fn()


if __name__ == "__main__":
  bench_one_op()
