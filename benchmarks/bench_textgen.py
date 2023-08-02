"""
Benchmark for text generation.
First come first serve.
Greedy generation, no sampling, no beam search.
"""

import dataclasses
import time

import numpy as np
import scipy.stats
import torch
from tqdm.auto import tqdm

from .benchmark_utils import gc_torch


@dataclasses.dataclass
class RequestSet:
  prompt_lens: np.ndarray
  output_lens: np.ndarray

  def __len__(self):
    return len(self.prompt_lens)


def generate_request_set(num_requests: int, maxlen: int) -> RequestSet:
  rng = np.random.Generator(np.random.PCG64(seed=0xabcdabcd987))
  prompt_dist = scipy.stats.lognorm(0.8, -1.0, 18.0)
  total_dist = scipy.stats.randint(0, maxlen)
  prompt_lens = np.minimum(
      prompt_dist.rvs(size=num_requests, random_state=rng),
      maxlen).astype(np.int32)
  total_lens = np.maximum(prompt_lens + 2,
                          total_dist.rvs(size=num_requests,
                                         random_state=rng)).astype(np.int32)
  output_lens = total_lens - prompt_lens
  return RequestSet(prompt_lens, output_lens)


@dataclasses.dataclass
class ModelConfig:
  num_layers: int
  num_heads: int
  hidden_size: int
  intermediate_size: int
  dtype: str
  device: str


@dataclasses.dataclass
class TextGenConfig:
  batch_size: int


@dataclasses.dataclass
class TextGenBenchResult:
  encode_latency: np.ndarray
  decode_latency: np.ndarray
  duration: float


@torch.inference_mode()
def textgen_punica(model_cfg: ModelConfig, textgen_cfg: TextGenConfig,
                   rs: RequestSet) -> TextGenBenchResult:
  from punica.models.llama import LlamaConfig, LlamaForCausalLM
  from punica.utils import CatTensor, KvPool
  gc_torch()
  torch.manual_seed(0xabcdabcd987)
  device = torch.device(model_cfg.device)
  dtype = getattr(torch, model_cfg.dtype)
  default_dtype = torch.get_default_dtype()
  torch.set_default_dtype(dtype)
  with device:
    model = LlamaForCausalLM(
        LlamaConfig(
            hidden_size=model_cfg.hidden_size,
            num_attention_heads=model_cfg.num_heads,
            intermediate_size=model_cfg.intermediate_size,
            num_hidden_layers=model_cfg.num_layers,
        )).to(device)
  torch.set_default_dtype(default_dtype)

  kvpool = KvPool(
      num_layers=model_cfg.num_layers,
      num_heads=model_cfg.num_heads,
      head_dim=model_cfg.hidden_size // model_cfg.num_heads,
      capacity=textgen_cfg.batch_size,
      block_len=2048,
      dtype=dtype,
      device=device)

  @dataclasses.dataclass
  class RequestContext:
    req_idx: int
    kv_idx: int
    pastlen: int
    output: list[int]

    encode_latency: float
    decode_start_at: float
    decode_latency: float

  t_start = time.perf_counter()
  next_req_idx = 0
  workset = []
  done = []
  pbar = tqdm(total=rs.output_lens.sum() + rs.prompt_lens.sum(), unit="token")
  while len(done) != len(rs):
    # Add new requests to workset
    while len(workset) < textgen_cfg.batch_size and next_req_idx < len(rs):
      idx = next_req_idx
      next_req_idx += 1
      prompt_len = rs.prompt_lens[idx]
      kv_idx = kvpool.alloc_block()
      prompt = torch.randint(
          0, 32000, (prompt_len,), dtype=torch.long, device=device)

      # Run encode separately because current implementation cannot mix
      # endcode and decode.
      t0 = time.perf_counter()
      past_lens = torch.tensor([0], dtype=torch.long, device=device)
      kvidx = torch.tensor([kv_idx], dtype=torch.long, device=device)
      logits, _h = model(kvpool, CatTensor(prompt, [prompt_len]), past_lens,
                         kvidx)
      logits = logits.cat
      next_token = torch.argmax(logits[-1, :], dim=-1).cpu().item()
      t1 = time.perf_counter()
      pbar.update(prompt_len + 1)

      workset.append(
          RequestContext(
              req_idx=idx,
              kv_idx=kv_idx,
              pastlen=logits.size(0),
              output=[next_token],
              encode_latency=t1 - t0,
              decode_start_at=t1,
              decode_latency=0.0,
          ))

    # Batched decode
    input_ids = CatTensor(
        torch.tensor([req.output[-1] for req in workset],
                     dtype=torch.long,
                     device=device), [1] * len(workset))
    past_lens = torch.tensor([req.pastlen for req in workset],
                             dtype=torch.long,
                             device=device)
    kvidx = torch.tensor([req.kv_idx for req in workset],
                         dtype=torch.long,
                         device=device)
    logits, _h = model(kvpool, input_ids, past_lens, kvidx)
    next_tokens = torch.argmax(logits.cat, dim=-1).cpu().numpy()
    assert len(next_tokens) == len(workset)
    pbar.update(len(workset))
    new_workset = []
    for batch_idx, logits in enumerate(logits.split()):
      req = workset[batch_idx]
      req.output.append(next_tokens[batch_idx])
      if len(req.output) == rs.output_lens[req.req_idx]:
        req.decode_latency = time.perf_counter() - req.decode_start_at
        kvpool.free_block(req.kv_idx)
        done.append(req)
      else:
        new_workset.append(req)
    workset = new_workset
  t_end = time.perf_counter()
  pbar.close()

  done.sort(key=lambda req: req.req_idx)
  return TextGenBenchResult(
      encode_latency=np.array([req.encode_latency for req in done]),
      decode_latency=np.array([req.decode_latency for req in done]),
      duration=t_end - t_start,
  )


def main():
  model_cfg = ModelConfig(
      num_layers=32,
      num_heads=32,
      hidden_size=4096,
      intermediate_size=11008,
      dtype="float16",
      device="cuda:0",
  )
  rs = generate_request_set(num_requests=50, maxlen=500)
  textgen_cfg = TextGenConfig(batch_size=8)
  res = textgen_punica(model_cfg, textgen_cfg, rs)

  t = res.encode_latency / rs.prompt_lens
  print("batch_size:", textgen_cfg.batch_size)
  print(
      "encode_latency:",
      f"{res.encode_latency.mean()*1e3:.3f}ms ± {res.encode_latency.std()*1e3:.3f}ms per request;",
      f"{t.mean()*1e3:.3f}ms ± {t.std()*1e3:.3f}ms per token")
  t = res.decode_latency / rs.output_lens
  print("decode_latency:",
        f"{t.mean()*1e3:.3f}ms ± {t.std()*1e3:.3f}ms per token")
  t = rs.output_lens / res.decode_latency
  print("decode_throughput:",
        f"{t.mean():.3f} tokens/s ± {t.std():.3f} tokens/s")
  print("duration:", f"{res.duration:.3f}s")


if __name__ == "__main__":
  main()
