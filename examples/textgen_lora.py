import argparse

import torch
import transformers

from punica import (
    BatchedKvCache,
    BatchedLlamaLoraWeight,
    BatchLenInfo,
    KvCache,
    KvPool,
    LlamaForCausalLMWithLora,
    LlamaLoraWeight,
)


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

        # Logits processing adapted from: https://github.com/lm-sys/FastChat/blob/bb7ca37c2bfad629ba4751dec188bdcdc2cf0c81/fastchat/serve/inference.py
        self.logits_processor = transformers.LogitsProcessorList()
        if temperature > 0 and temperature != 1.0:
            self.logits_processor.append(
                transformers.TemperatureLogitsWarper(temperature)
            )
        if repetition_penalty > 1.0:
            self.logits_processor.append(
                transformers.RepetitionPenaltyLogitsProcessor(repetition_penalty)
            )
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
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--lora-weight", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-new-tokens", type=int, default=500)

    args = parser.parse_args()
    dtype = torch.float16
    device = torch.device("cuda:0")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model, use_fast=True
    )
    model_config = transformers.LlamaConfig.from_pretrained(args.base_model)
    model = LlamaForCausalLMWithLora.from_pretrained(
        args.base_model, low_cpu_mem_usage=True, torch_dtype=dtype
    ).to(device)
    kvpool = KvPool(
        num_layers=model_config.num_hidden_layers,
        num_heads=model_config.num_attention_heads,
        head_dim=model_config.hidden_size // model_config.num_attention_heads,
        page_len=16,
        dtype=dtype,
        device=device,
    )
    tmp = torch.load(args.lora_weight, map_location=device, weights_only=True)
    lora_rank = tmp["q.A"].size(1)
    lora_weight = LlamaLoraWeight(model_config, lora_rank, dtype, device)
    lora_weight.copy_from_tensors(tmp)
    del tmp

    input_ids = tokenizer.encode(args.prompt)
    kvcache = KvCache(kvpool, len(input_ids))
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
    print(text, end="", flush=True)
    last_print_len = len(text)

    # Prefill
    logits, _ = model(
        input_ids=torch.tensor(input_ids, dtype=torch.long, device=device),
        blen=BatchLenInfo([len(input_ids)], 0, device),
        prefill_kv=BatchedKvCache([kvcache]),
        decode_kv=None,
        lora=BatchedLlamaLoraWeight([lora_weight], [len(input_ids)]),
    )
    next_token_id = textgen.get_next_token_id(logits)
    textgen.append_token(next_token_id)

    # Decode
    while not textgen.is_stop():
        kvcache.acquire_one()
        logits, _ = model(
            input_ids=torch.tensor([next_token_id], dtype=torch.long, device=device),
            blen=BatchLenInfo([], 1, device),
            prefill_kv=None,
            decode_kv=BatchedKvCache([kvcache]),
            lora=BatchedLlamaLoraWeight([lora_weight], [1]),
        )
        next_token_id = textgen.get_next_token_id(logits)
        textgen.append_token(next_token_id)

        text = tokenizer.decode(
            textgen.output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        print(text[last_print_len:], end="", flush=True)
        last_print_len = len(text)

    kvcache.release()
    print()


if __name__ == "__main__":
    main()
