import argparse
import itertools

import torch

import punica.ops
import punica.ops._kernels

from .benchmark_utils import bench, gc_torch


class mha_decode_Resources:

  def __init__(
      self,
      num_heads: int,
      head_dim: int,
      num_layers: int,
      maxlen: int,
      batch_size: int,
      past_lens: int | torch.Tensor,
      dtype: str,
      device: torch.device,
  ):
    dtype = getattr(torch, dtype)
    self.kvbuf = torch.randn(
        (batch_size, num_layers, 2, maxlen, num_heads, head_dim),
        dtype=dtype,
        device=device)
    self.q_proj = torch.randn((batch_size, num_heads, head_dim),
                              dtype=dtype,
                              device=device)
    self.k_proj = torch.randn((batch_size, num_heads, head_dim),
                              dtype=dtype,
                              device=device)
    self.v_proj = torch.randn((batch_size, num_heads, head_dim),
                              dtype=dtype,
                              device=device)
    self.kvidx = torch.arange(0, batch_size, dtype=torch.long, device=device)

    if isinstance(past_lens, int):
      self.past_lens = torch.ones(
          (batch_size,), dtype=torch.long, device=device) * past_lens
    elif isinstance(past_lens, torch.Tensor):
      self.past_lens = past_lens
    else:
      raise ValueError("Unknown type for `past_lens`.", past_lens)


def bench_rotary_mha_decode_fixed_length():
  model_sizes = [
      # Regular sizes
      (12, 64, 12, 2048, "float16"),
      (16, 64, 24, 2048, "float16"),
      (32, 64, 24, 2048, "float16"),
      (32, 80, 32, 2048, "float16"),
      (32, 128, 32, 2048, "float16"),
      (40, 128, 40, 2048, "float16"),
      (56, 128, 48, 2048, "float16"),
      (72, 128, 64, 2048, "float16"),
      # (96, 128, 96, 2048, "float16"),

      # Irregular sizes
      (32, 128, 32, 3333, "float16"),
      (32, 128, 13, 2048, "float16"),
      (32, 96, 32, 2048, "float16"),
      (13, 128, 32, 2048, "float16"),
      (13, 64, 17, 3333, "float16"),
  ]
  batch_sizes = list(range(1, 17))
  device = torch.device("cuda:0")

  print("bench_rotary_mha_decode_fixed_length")
  for (num_heads, head_dim, num_layers, maxlen,
       dtype_str), batch_size in itertools.product(model_sizes, batch_sizes):
    torch.manual_seed(0xabcdabcd987)
    past_lens = 10
    layer_idx = 0
    outputs = [
        f"n={num_heads}",
        f"d={head_dim:3d}",
        f"l={num_layers}",
        f"maxlen={maxlen}",
        f"{dtype_str}",
        f"bs={batch_size:2d}",
    ]
    try:
      gc_torch()
      t = mha_decode_Resources(
          num_heads=num_heads,
          head_dim=head_dim,
          num_layers=num_layers,
          maxlen=maxlen,
          batch_size=batch_size,
          past_lens=past_lens,
          dtype=dtype_str,
          device=device,
      )
      result = bench(lambda: punica.ops.rotary_mha_decode(
          t.q_proj, t.k_proj, t.v_proj, t.past_lens, t.kvbuf, t.kvidx, layer_idx
      ))
      outputs.append(f"{result.avg()*1e6:3.0f}us±{result.std()*1e6:3.0f}us")
    except torch.cuda.OutOfMemoryError:
      outputs.append("OOM")

    print(" | ".join(outputs))


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
