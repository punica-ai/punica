"""
Benchmark for text generation with LoRA.
Each sequence within a batch requests different LoRA weights.
First come first serve.
Greedy generation, no sampling, no beam search.
"""

import argparse
import dataclasses
import logging
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from .bench_textgen import (ModelConfig, RequestSet, TextGenBenchResult,
                            TextGenConfig, generate_request_set)


@dataclasses.dataclass
class LoraConfig:
  rank: int
  alpha: int


@torch.inference_mode()
def lora_punica(model_cfg: ModelConfig, lora_cfg: LoraConfig,
                textgen_cfg: TextGenConfig,
                rs: RequestSet) -> TextGenBenchResult:
  from punica.models.llama_lora import LlamaConfig, LlamaForCausalLMWithLora
  from punica.utils import (CatTensor, KvPool, LlamaLoraManager,
                            LlamaLoraModelWeightIndicies)
  torch.manual_seed(0xabcdabcd987)
  device = torch.device(model_cfg.device)
  dtype = getattr(torch, model_cfg.dtype)
  default_dtype = torch.get_default_dtype()
  torch.set_default_dtype(dtype)
  with device:
    model = LlamaForCausalLMWithLora(
        config=LlamaConfig(
            hidden_size=model_cfg.hidden_size,
            num_attention_heads=model_cfg.num_heads,
            intermediate_size=model_cfg.intermediate_size,
            num_hidden_layers=model_cfg.num_layers,
        ),
        lora_scale=1.0,
    ).to(device)
  torch.set_default_dtype(default_dtype)

  kvpool = KvPool(
      num_layers=model_cfg.num_layers,
      num_heads=model_cfg.num_heads,
      head_dim=model_cfg.hidden_size // model_cfg.num_heads,
      capacity=textgen_cfg.batch_size,
      block_len=2048,
      dtype=dtype,
      device=device)
  lora_mgr = LlamaLoraManager(
      capacity=textgen_cfg.batch_size,
      num_layers=model_cfg.num_layers,
      hidden_size=model_cfg.hidden_size,
      intermediate_size=model_cfg.intermediate_size,
      lora_rank=lora_cfg.rank,
      dtype=dtype,
      device=device,
  )
  for mgr in [lora_mgr.mgr_hh, lora_mgr.mgr_hi, lora_mgr.mgr_ih]:
    mgr._wa_T.copy_(torch.randn_like(mgr._wa_T))
    mgr._wb_T.copy_(torch.randn_like(mgr._wb_T))
  lora_models = [lora_mgr.alloc() for _ in range(textgen_cfg.batch_size)]

  @dataclasses.dataclass
  class RequestContext:
    req_idx: int
    kv_idx: int
    pastlen: int
    output: list[int]

    encode_latency: float
    decode_start_at: float
    decode_latency: float

  next_req_idx = 0
  workset = []
  done = []
  pbar = tqdm(
      total=rs.output_lens.sum() + rs.prompt_lens.sum(),
      unit="token",
      desc="punica")
  t_start = time.perf_counter()
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
      # encode and decode.
      t0 = time.perf_counter()
      past_lens = torch.tensor([0], dtype=torch.long, device=device)
      kvidx = torch.tensor([kv_idx], dtype=torch.long, device=device)
      lora = LlamaLoraModelWeightIndicies(
          [lora_models[idx % len(lora_models)]] * prompt_len)
      logits, _h = model(kvpool, CatTensor(prompt, [prompt_len]), past_lens,
                         kvidx, lora)
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
    lora = LlamaLoraModelWeightIndicies(
        [lora_models[req.req_idx % len(lora_models)] for req in workset])
    logits, _h = model(kvpool, input_ids, past_lens, kvidx, lora)
    next_tokens = torch.argmax(logits.cat, dim=-1).cpu().numpy()
    assert len(next_tokens) == len(workset)
    pbar.update(len(workset))
    new_workset = []
    for batch_idx in range(len(workset)):
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


def _lora_hf_bs1_internal(model_cfg: ModelConfig, lora_cfg: LoraConfig,
                          textgen_cfg: TextGenConfig, rs: RequestSet,
                          use_deepspeed: bool) -> TextGenBenchResult:
  import peft
  from transformers import LlamaConfig, LlamaForCausalLM

  if textgen_cfg.batch_size != 1:
    raise ValueError("batch_size must be 1.")

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
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            inference_mode=True,
            r=lora_cfg.rank,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=0.0,
            bias="none",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                "down_proj"
            ]))
  torch.set_default_dtype(default_dtype)

  if use_deepspeed:
    import deepspeed
    ds_engine = deepspeed.init_inference(
        model=model, max_out_tokens=2048, replace_with_kernel_inject=True)
    model = ds_engine.module

  pbar = tqdm(
      total=rs.output_lens.sum() + rs.prompt_lens.sum(),
      unit="token",
      desc="hf_bs1" if not use_deepspeed else "ds_bs1")
  model_kwargs = dict(
      use_cache=True,
      output_attentions=False,
      output_hidden_states=False,
      return_dict=True,
  )
  encode_latency = []
  decode_latency = []
  t_start = time.perf_counter()
  for idx in range(len(rs)):
    # Encode
    input_ids = [
        torch.randint(
            0, 32000, (rs.prompt_lens[idx],), dtype=torch.long, device=device)
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=0.0)
    t0 = time.perf_counter()
    ret = model(input_ids=input_ids, **model_kwargs)
    past_key_values = ret.past_key_values
    input_ids = torch.argmax(ret.logits[:, -1, :], dim=-1, keepdim=True)
    outputs = [input_ids.cpu().item()]
    t1 = time.perf_counter()
    pbar.update(rs.prompt_lens[idx] + 1)
    encode_latency.append(t1 - t0)

    # Decode
    t2 = time.perf_counter()
    for _ in range(rs.output_lens[idx] - 1):
      ret = model(
          input_ids=input_ids, past_key_values=past_key_values, **model_kwargs)
      past_key_values = ret.past_key_values
      input_ids = torch.argmax(ret.logits, dim=-1)
      next_token = input_ids.cpu().item()
      outputs.append(next_token)
      pbar.update()
    decode_latency.append(time.perf_counter() - t2)
  t_end = time.perf_counter()
  pbar.close()

  return TextGenBenchResult(
      encode_latency=np.array(encode_latency),
      decode_latency=np.array(decode_latency),
      duration=t_end - t_start,
  )


@torch.inference_mode()
def lora_hf_bs1(model_cfg: ModelConfig, lora_cfg: LoraConfig,
                textgen_cfg: TextGenConfig,
                rs: RequestSet) -> TextGenBenchResult:
  return _lora_hf_bs1_internal(
      model_cfg, lora_cfg, textgen_cfg, rs, use_deepspeed=False)


@torch.inference_mode()
def lora_ds_bs1(model_cfg: ModelConfig, lora_cfg: LoraConfig,
                textgen_cfg: TextGenConfig,
                rs: RequestSet) -> TextGenBenchResult:
  return _lora_hf_bs1_internal(
      model_cfg, lora_cfg, textgen_cfg, rs, use_deepspeed=True)


BENCH_FN = {
    "punica": lora_punica,
    "hf_bs1": lora_hf_bs1,
    "ds_bs1": lora_ds_bs1,
}

MODEL_CFGS = {
    "7b":
        ModelConfig(
            num_layers=32,
            num_heads=32,
            hidden_size=4096,
            intermediate_size=11008,
        ),
    "13b":
        ModelConfig(
            num_layers=40,
            num_heads=40,
            hidden_size=5120,
            intermediate_size=13824,
        ),
}


def bench_one():
  parser = argparse.ArgumentParser()
  parser.add_argument("--system", choices=BENCH_FN.keys(), default="punica")
  parser.add_argument("--model", choices=MODEL_CFGS.keys(), default="7b")
  parser.add_argument("--lora-rank", type=int, default=16)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--num-batches", type=int, default=10)
  parser.add_argument("--maxlen", type=int, default=2048)
  parser.add_argument(
      "--dtype", choices=["float16", "bfloat16"], default="float16")
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  bench_fn = BENCH_FN[args.system]
  model_cfg = MODEL_CFGS[args.model]
  model_cfg.dtype = args.dtype
  model_cfg.device = args.device

  rs = generate_request_set(
      num_requests=args.batch_size * args.num_batches, maxlen=args.maxlen)
  textgen_cfg = TextGenConfig(batch_size=args.batch_size)
  lora_cfg = LoraConfig(rank=args.lora_rank, alpha=args.lora_rank * 2)
  res = bench_fn(model_cfg, lora_cfg, textgen_cfg, rs)

  t = res.encode_latency / rs.prompt_lens
  print("num_requests:", len(rs))
  print("batch_size:", textgen_cfg.batch_size)
  print(
      "encode_latency:",
      f"{res.encode_latency.mean()*1e3:.3f}ms ± {res.encode_latency.std()*1e3:.3f}ms per request;",
      f"{t.mean()*1e3:.3f}ms ± {t.std()*1e3:.3f}ms per token")
  t = res.decode_latency / rs.output_lens
  print("decode_latency:",
        f"{t.mean()*1e3:.3f}ms ± {t.std()*1e3:.3f}ms per token")
  print("total prompt tokens:", rs.prompt_lens.sum())
  print("total new tokens:", rs.output_lens.sum())
  print("duration:", f"{res.duration:.3f}s")
  print(
      "throughput ((prompt+new)/duration):",
      f"{(rs.prompt_lens.sum()+rs.output_lens.sum())/res.duration:.3f} token/s")


if __name__ == "__main__":
  bench_one()
