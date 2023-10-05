import argparse

import torch
import transformers
import torch.distributed as dist

import punica
from punica.utils import create_tensor_parallel_group, init_distributed, is_distributed, get_local_rank_from_launcher
from punica.models.tensor_parallel import AutoTP


class TextGeneration:

  def __init__(
      self,
      input_ids: list[int],
      *,
      temperature: float,
      repetition_penalty: float,
      top_p: float,
      top_k: int,
      max_new_tokens: int,
      stop_token_id: int,
  ):
    self.temperature = temperature
    self.repetition_penalty = repetition_penalty
    self.top_p = top_p
    self.top_k = top_k
    self.max_new_tokens = max_new_tokens
    self.stop_token_id = stop_token_id

    self.logits_processor = transformers.LogitsProcessorList()
    if temperature > 0 and temperature != 1.0:
      self.logits_processor.append(
          transformers.TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
      self.logits_processor.append(
          transformers.RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 0 < top_p < 1.0:
      self.logits_processor.append(transformers.TopPLogitsWarper(top_p))
    if top_k > 0:
      self.logits_processor.append(transformers.TopKLogitsWarper(top_k))

    self.output_ids = [int(x) for x in input_ids]
    self.prompt_len = len(self.output_ids)

  def get_next_token_id(self, logits: torch.Tensor) -> int:
    if self.logits_processor:
      if self.repetition_penalty > 1.0:
        t = torch.as_tensor([self.output_ids], device=logits.device)
      else:
        t = None
      last_token_logits = self.logits_processor(t, logits[-1].unsqueeze(0))[0]
    else:
      last_token_logits = logits[-1, :]

    if self.temperature <= 0 or self.top_p <= 0:
      _, indices = torch.topk(last_token_logits, 2)
    else:
      probs = torch.softmax(last_token_logits, dim=-1)
      indices = torch.multinomial(probs, num_samples=2)
    token = int(indices.tolist()[0])
    return token

  def append_token(self, token_id: int):
    self.output_ids.append(token_id)

  def is_stop(self) -> int:
    if len(self.output_ids) - self.prompt_len >= self.max_new_tokens:
      return True
    if self.output_ids[-1] == self.stop_token_id:
      return True
    return False


@torch.inference_mode()
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
  parser.add_argument(
      "--prompt", default="Give me a 3 day travel plan for Seattle.")
  parser.add_argument(
      "--template",
      default="[INST] <<SYS>> You are a helpful, respectful and honest assistant. <</SYS>>\n{prompt} [/INST]\n"
  )
  parser.add_argument("--temperature", type=float, default=0.7)
  parser.add_argument("--repetition-penalty", type=float, default=1.1)
  parser.add_argument("--top-p", type=float, default=0.9)
  parser.add_argument("--top-k", type=int, default=-1)
  parser.add_argument("--max-new-tokens", type=int, default=2000)

  args = parser.parse_args()
  dtype = torch.float16
  local_rank = get_local_rank_from_launcher()
  torch.cuda.set_device(local_rank)
  device = torch.device("cuda:{}".format(local_rank))

  tokenizer = transformers.AutoTokenizer.from_pretrained(
      args.model, use_fast=True)
  model_config = transformers.LlamaConfig.from_pretrained(args.model)
  model = punica.models.llama.LlamaForCausalLM.from_pretrained(
      args.model, low_cpu_mem_usage=True, torch_dtype=dtype)
  if is_distributed():
    init_distributed()
    mp_group = create_tensor_parallel_group()
    mp_size = len(dist.get_process_group_ranks(mp_group))
    auto_tp = AutoTP(mp_group, mp_size)
    model = auto_tp.replace_module(model)
  else:
    mp_size = 1
  model = model.to(device)
  
  kvpool = punica.utils.KvPool(
      num_layers=model_config.num_hidden_layers,
      num_heads=model_config.num_attention_heads // mp_size,
      head_dim=model_config.hidden_size // model_config.num_attention_heads,
      capacity=4096 // 16,
      block_len=16,
      dtype=dtype,
      device=device,
  )

  prompt = args.template.format(prompt=args.prompt)
  input_ids = tokenizer.encode(prompt)
  kvcache = punica.utils.KvCache(kvpool, len(input_ids))
  textgen = TextGeneration(
      input_ids,
      temperature=args.temperature,
      repetition_penalty=args.repetition_penalty,
      top_p=args.top_p,
      top_k=args.top_k,
      max_new_tokens=args.max_new_tokens,
      stop_token_id=tokenizer.eos_token_id,
  )

  # Print prompt
  text = tokenizer.decode(
      input_ids,
      skip_special_tokens=True,
      spaces_between_special_tokens=False,
      clean_up_tokenization_spaces=True,
  )
  if mp_size == 1 or dist.get_rank() == 0:
    print(text, end="", flush=True)
  last_print_len = len(text)

  # Prefill
  logits, _ = model(
      input_ids=torch.tensor(input_ids, dtype=torch.long, device=device),
      blen=punica.utils.BatchLenInfo([len(input_ids)], 0, device),
      prefill_kv=punica.utils.BatchedKvCache([kvcache]),
      decode_kv=None,
  )
  next_token_id = textgen.get_next_token_id(logits)
  textgen.append_token(next_token_id)

  if mp_size == 1 or dist.get_rank() == 0:
    print("Prefill finished")
  # Decode
  while not textgen.is_stop():
    kvcache.acquire_one()
    logits, _ = model(
        input_ids=torch.tensor([next_token_id], dtype=torch.long,
                               device=device),
        blen=punica.utils.BatchLenInfo([], 1, device),
        prefill_kv=None,
        decode_kv=punica.utils.BatchedKvCache([kvcache]),
    )
    next_token_id = textgen.get_next_token_id(logits)
    textgen.append_token(next_token_id)

    text = tokenizer.decode(
        textgen.output_ids,
        skip_special_tokens=True,
        spaces_between_special_tokens=False,
        clean_up_tokenization_spaces=True,
    )
    if mp_size == 1 or dist.get_rank() == 0:
      print(text[last_print_len:], end="", flush=True)
    last_print_len = len(text)

  if mp_size == 1 or dist.get_rank() == 0:
    print()


if __name__ == "__main__":
  main()
