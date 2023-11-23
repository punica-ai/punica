"""
Benchmark for text generation.
First come first serve.
Greedy generation, no sampling, no beam search.
"""

import argparse
import dataclasses
import logging
import time

import numpy as np
import scipy.stats
import torch
from tqdm.auto import tqdm

from .benchmark_utils import batched


@dataclasses.dataclass
class RequestSet:
    prompt_lens: np.ndarray
    output_lens: np.ndarray

    def __len__(self):
        return len(self.prompt_lens)


def generate_request_set(num_requests: int, maxlen: int) -> RequestSet:
    rng = np.random.Generator(np.random.PCG64(seed=0xABCDABCD987))
    prompt_dist = scipy.stats.lognorm(0.8, -1.0, 18.0)
    total_dist = scipy.stats.randint(0, maxlen)
    prompt_lens = []
    output_lens = []
    for _ in range(num_requests):
        p = min(max(1, prompt_dist.rvs(random_state=rng)), maxlen - 2)
        t = max(p + 2, total_dist.rvs(random_state=rng))
        prompt_lens.append(p)
        output_lens.append(t - p)
    return RequestSet(
        np.array(prompt_lens, dtype=np.int32),
        np.array(output_lens, dtype=np.int32),
    )


@dataclasses.dataclass
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    hidden_size: int
    intermediate_size: int
    dtype: str = dataclasses.field(default="float16")
    device: str = dataclasses.field(default="cuda:0")


@dataclasses.dataclass
class TextGenConfig:
    batch_size: int


@dataclasses.dataclass
class TextGenBenchResult:
    encode_latency: np.ndarray
    decode_latency: np.ndarray
    duration: float


@torch.inference_mode()
def textgen_punica(
    model_cfg: ModelConfig, textgen_cfg: TextGenConfig, rs: RequestSet
) -> TextGenBenchResult:
    from transformers import LlamaConfig

    from punica import BatchedKvCache, BatchLenInfo, KvCache, KvPool, LlamaForCausalLM

    torch.manual_seed(0xABCDABCD987)
    device = torch.device(model_cfg.device)
    dtype = getattr(torch, model_cfg.dtype)
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    with device:
        model = LlamaForCausalLM(
            LlamaConfig(
                hidden_size=model_cfg.hidden_size,
                num_attention_heads=model_cfg.num_qo_heads,
                num_key_value_heads=model_cfg.num_kv_heads,
                intermediate_size=model_cfg.intermediate_size,
                num_hidden_layers=model_cfg.num_layers,
            )
        ).to(device)
    torch.set_default_dtype(default_dtype)

    kvpool = KvPool(
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_kv_heads,
        head_dim=model_cfg.hidden_size // model_cfg.num_qo_heads,
        page_len=16,
        dtype=dtype,
        device=device,
    )

    @dataclasses.dataclass
    class RequestContext:
        req_idx: int
        kvcache: KvCache
        output: list[int]

        encode_latency: float
        decode_start_at: float
        decode_latency: float

    next_req_idx = 0
    workset: list[RequestContext] = []
    done = []
    pbar = tqdm(
        total=rs.output_lens.sum() + rs.prompt_lens.sum(), unit="token", desc="punica"
    )
    t_start = time.perf_counter()
    while len(done) != len(rs):
        # Add at most one new request to workset
        newreqs = []
        if len(workset) + len(newreqs) < textgen_cfg.batch_size and next_req_idx < len(
            rs
        ):
            idx = next_req_idx
            next_req_idx += 1
            prompt_len = rs.prompt_lens[idx]
            kvcache = KvCache(kvpool, prompt_len)
            prompt = torch.randint(0, 32000, (prompt_len,), device="cpu").tolist()
            newreqs.append((idx, prompt, kvcache))

        # Batched prefill and decode
        t1 = time.perf_counter()
        input_ids = []
        prefill_kv = []
        for _, prompt, kvcache in newreqs:
            input_ids.extend(prompt)
            prefill_kv.append(kvcache)
        input_ids.extend(req.output[-1] for req in workset)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        blen = BatchLenInfo(
            prefills=[len(prompt) for _, prompt, _ in newreqs],
            decode=len(workset),
            indptr_device=device,
        )
        prefill_kv = BatchedKvCache(prefill_kv) if newreqs else None
        decode_kv = (
            BatchedKvCache([req.kvcache for req in workset]) if workset else None
        )
        logits, _ = model(input_ids, blen, prefill_kv, decode_kv)
        t2 = time.perf_counter()

        # Post-process prefill
        new_workset: list[RequestContext] = []
        if newreqs:
            last_token_indicies = blen.indptr[1:] - 1
            next_tokens = (
                torch.argmax(logits[last_token_indicies], dim=-1).cpu().numpy()
            )
            assert len(next_tokens) == len(newreqs)
            pbar.update(blen.doff + len(newreqs))
            for batch_idx in range(len(newreqs)):
                req_idx, prompt, kvcache = newreqs[batch_idx]
                kvcache.acquire_one()
                req = RequestContext(
                    req_idx=req_idx,
                    kvcache=kvcache,
                    output=[next_tokens[batch_idx]],
                    encode_latency=t2 - t1,
                    decode_start_at=t1,
                    decode_latency=0.0,
                )
                new_workset.append(req)

        # Post-process decode
        if workset:
            next_tokens = torch.argmax(logits[blen.doff :], dim=-1).cpu().numpy()
            assert len(next_tokens) == len(workset)
            pbar.update(len(workset))
            for batch_idx in range(len(workset)):
                req = workset[batch_idx]
                req.output.append(next_tokens[batch_idx])
                if len(req.output) == rs.output_lens[req.req_idx]:
                    req.decode_latency = t2 - req.decode_start_at
                    req.kvcache.release()
                    done.append(req)
                else:
                    new_workset.append(req)
                    req.kvcache.acquire_one()

        workset = new_workset
    t_end = time.perf_counter()
    pbar.close()

    done.sort(key=lambda req: req.req_idx)
    return TextGenBenchResult(
        encode_latency=np.array([req.encode_latency for req in done]),
        decode_latency=np.array([req.decode_latency for req in done]),
        duration=t_end - t_start,
    )


def _textgen_hf_pad_internal(
    model_cfg: ModelConfig,
    textgen_cfg: TextGenConfig,
    rs: RequestSet,
    use_deepspeed: bool,
) -> TextGenBenchResult:
    """
    Requests in one batch remain in the same batch for the entire encode
    and decode process.
    Pad requests to the longest length in the batch.
    Repeat decode until all requests are done.
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0xABCDABCD987)
    device = torch.device(model_cfg.device)
    dtype = getattr(torch, model_cfg.dtype)
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    with device:
        model = LlamaForCausalLM(
            LlamaConfig(
                hidden_size=model_cfg.hidden_size,
                num_attention_heads=model_cfg.num_qo_heads,
                num_key_value_heads=model_cfg.num_kv_heads,
                intermediate_size=model_cfg.intermediate_size,
                num_hidden_layers=model_cfg.num_layers,
            )
        ).to(device)
    torch.set_default_dtype(default_dtype)
    if use_deepspeed:
        import deepspeed

        ds_engine = deepspeed.init_inference(
            model=model, max_out_tokens=2048, replace_with_kernel_inject=True
        )
        model = ds_engine.module

    pbar = tqdm(
        total=rs.output_lens.sum() + rs.prompt_lens.sum(),
        unit="token",
        desc="hf_pad" if not use_deepspeed else "ds_pad",
    )
    model_kwargs = dict(
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    encode_latency = []
    decode_latency = []
    t_start = time.perf_counter()
    for req_indicies in batched(range(len(rs)), textgen_cfg.batch_size):
        bs = len(req_indicies)
        # Encode
        input_ids = [
            torch.randint(
                0, 32000, (rs.prompt_lens[idx],), dtype=torch.long, device=device
            )
            for idx in req_indicies
        ]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=0.0
        )
        t0 = time.perf_counter()
        ret = model(input_ids=input_ids, **model_kwargs)
        past_key_values = ret.past_key_values
        input_ids = torch.argmax(ret.logits[:, -1, :], dim=-1, keepdim=True)
        outputs = [[t] for t in input_ids.view(bs).cpu().numpy()]
        t1 = time.perf_counter()
        pbar.update(sum(rs.prompt_lens[idx] + 1 for idx in req_indicies))
        for _ in range(len(req_indicies)):
            encode_latency.append(t1 - t0)

        # Decode
        dl = [None] * bs
        t2 = time.perf_counter()
        for _ in range(max(rs.output_lens[idx] - 1 for idx in req_indicies)):
            ret = model(
                input_ids=input_ids, past_key_values=past_key_values, **model_kwargs
            )
            past_key_values = ret.past_key_values
            input_ids = torch.argmax(ret.logits, dim=-1)
            next_tokens = input_ids.view(bs).cpu().numpy()
            for i in range(bs):
                outlen = rs.output_lens[req_indicies[i]]
                if len(outputs[i]) < outlen:
                    outputs[i].append(next_tokens[i])
                    pbar.update()
                    if len(outputs[i]) == outlen:
                        dl[i] = time.perf_counter() - t2
        decode_latency.extend(dl)
    t_end = time.perf_counter()
    pbar.close()

    return TextGenBenchResult(
        encode_latency=np.array(encode_latency),
        decode_latency=np.array(decode_latency),
        duration=t_end - t_start,
    )


@torch.inference_mode()
def textgen_hf_pad(
    model_cfg: ModelConfig, textgen_cfg: TextGenConfig, rs: RequestSet
) -> TextGenBenchResult:
    return _textgen_hf_pad_internal(model_cfg, textgen_cfg, rs, use_deepspeed=False)


@torch.inference_mode()
def textgen_ds_pad(
    model_cfg: ModelConfig, textgen_cfg: TextGenConfig, rs: RequestSet
) -> TextGenBenchResult:
    return _textgen_hf_pad_internal(model_cfg, textgen_cfg, rs, use_deepspeed=True)


@torch.inference_mode()
def textgen_ft_pad(
    model_cfg: ModelConfig, textgen_cfg: TextGenConfig, rs: RequestSet
) -> TextGenBenchResult:
    from .fastertransformer import build_ext

    ft = build_ext()
    rng = np.random.Generator(np.random.PCG64(seed=0xABCDABCD987))
    if model_cfg.num_qo_heads != model_cfg.num_kv_heads:
        raise ValueError("GQA is not supported for FasterTransformer")
    model = ft.FtLlama(
        num_heads=model_cfg.num_qo_heads,
        head_dim=model_cfg.hidden_size // model_cfg.num_qo_heads,
        inter_size=model_cfg.intermediate_size,
        num_layers=model_cfg.num_layers,
        dtype=model_cfg.dtype,
        device_id=torch.device(model_cfg.device).index,
    )

    pbar = tqdm(
        total=rs.output_lens.sum() + rs.prompt_lens.sum(), unit="token", desc="ft_pad"
    )
    encode_latency = []
    decode_latency = []
    t_start = time.perf_counter()
    for req_indicies in batched(range(len(rs)), textgen_cfg.batch_size):
        bs = len(req_indicies)

        # Decode Callback
        outlens = [0] * bs
        dl = [None] * bs
        t_decode_start = None

        def decode_cb():
            nonlocal t_decode_start
            t3 = time.perf_counter()
            if t_decode_start is None:
                t_decode_start = t3
                pbar.update(sum(rs.prompt_lens[idx] + 1 for idx in req_indicies))

            for i in range(bs):
                target_len = rs.output_lens[req_indicies[i]]
                if outlens[i] < target_len:
                    outlens[i] += 1
                    pbar.update()
                    if outlens[i] == target_len:
                        dl[i] = t3 - t_decode_start

        # Encode and decode all
        input_ids = [
            rng.integers(0, 32000, (rs.prompt_lens[idx],), dtype=int).tolist()
            for idx in req_indicies
        ]
        max_output_lens = max(rs.output_lens[idx] for idx in req_indicies)
        t0 = time.perf_counter()
        model.forward(input_ids, max_output_lens, decode_cb)
        for _ in range(len(req_indicies)):
            encode_latency.append(t_decode_start - t0)
        decode_latency.extend(dl)
    t_end = time.perf_counter()
    pbar.close()

    return TextGenBenchResult(
        encode_latency=np.array(encode_latency),
        decode_latency=np.array(decode_latency),
        duration=t_end - t_start,
    )


@torch.inference_mode()
def textgen_vllm(
    model_cfg: ModelConfig, textgen_cfg: TextGenConfig, rs: RequestSet
) -> TextGenBenchResult:
    import vllm.transformers_utils.config
    from transformers import LlamaConfig
    from vllm import EngineArgs, LLMEngine, SamplingParams

    class FakeModelConfig(LlamaConfig):
        def __init__(self, **kwargs):
            kwargs["num_hidden_layers"] = model_cfg.num_layers
            kwargs["num_attention_heads"] = model_cfg.num_qo_heads
            kwargs["num_kv_attention_heads"] = model_cfg.num_kv_heads
            kwargs["hidden_size"] = model_cfg.hidden_size
            kwargs["intermediate_size"] = model_cfg.intermediate_size
            kwargs["torch_dtype"] = model_cfg.dtype
            super().__init__(**kwargs)

    vllm.transformers_utils.config._CONFIG_REGISTRY["llama"] = FakeModelConfig
    logging.getLogger("vllm").setLevel(logging.WARNING)
    rng = np.random.Generator(np.random.PCG64(seed=0xABCDABCD987))
    llm_engine = LLMEngine.from_engine_args(
        EngineArgs(
            model="decapoda-research/llama-13b-hf",
            tokenizer="hf-internal-testing/llama-tokenizer",
            dtype="float16",
            load_format="dummy",
        )
    )

    @dataclasses.dataclass
    class RequestContext:
        req_idx: int

        encode_start_at: float
        encode_latency: float
        decode_start_at: float
        decode_latency: float

    next_req_idx = 0
    workset = {}
    done = []
    pbar = tqdm(
        total=rs.output_lens.sum() + rs.prompt_lens.sum(), unit="token", desc="vllm"
    )

    t_start = time.perf_counter()
    while len(done) != len(rs):
        # Add new requests to workset
        while len(workset) < textgen_cfg.batch_size and next_req_idx < len(rs):
            idx = next_req_idx
            next_req_idx += 1
            prompt_ids = rng.integers(
                0, 32000, (rs.prompt_lens[idx],), dtype=np.int32
            ).tolist()
            sampling_params = SamplingParams(
                n=1, temperature=0.0, ignore_eos=True, max_tokens=rs.output_lens[idx]
            )
            t0 = time.perf_counter()
            reqid = str(idx)
            llm_engine.add_request(
                reqid,
                prompt=None,
                sampling_params=sampling_params,
                prompt_token_ids=prompt_ids,
            )
            workset[reqid] = RequestContext(
                req_idx=idx,
                encode_start_at=t0,
                encode_latency=0.0,
                decode_start_at=0.0,
                decode_latency=0.0,
            )

        # Run vLLM
        step_outputs = llm_engine.step()
        t1 = time.perf_counter()
        for out in step_outputs:
            ctx = workset[out.request_id]
            out_tokens = out.outputs[0].token_ids
            if len(out_tokens) == 1:
                ctx.encode_latency = t1 - ctx.encode_start_at
                ctx.decode_start_at = t1
                pbar.update(rs.prompt_lens[ctx.req_idx] + 1)
            else:
                pbar.update(1)

            if out.finished:
                assert len(out_tokens) == rs.output_lens[ctx.req_idx]
            if len(out_tokens) == rs.output_lens[ctx.req_idx]:
                assert out.finished
                ctx.decode_latency = t1 - ctx.decode_start_at
                done.append(ctx)
                del workset[out.request_id]
    t_end = time.perf_counter()
    pbar.close()

    done.sort(key=lambda req: req.req_idx)
    return TextGenBenchResult(
        encode_latency=np.array([req.encode_latency for req in done]),
        decode_latency=np.array([req.decode_latency for req in done]),
        duration=t_end - t_start,
    )


BENCH_FN = {
    "punica": textgen_punica,
    "vllm": textgen_vllm,
    "hf_pad": textgen_hf_pad,
    "ds_pad": textgen_ds_pad,
    "ft_pad": textgen_ft_pad,
}

MODEL_CFGS = {
    "7b": ModelConfig(
        num_layers=32,
        num_qo_heads=32,
        num_kv_heads=32,
        hidden_size=4096,
        intermediate_size=11008,
    ),
    "13b": ModelConfig(
        num_layers=40,
        num_qo_heads=40,
        num_kv_heads=40,
        hidden_size=5120,
        intermediate_size=13824,
    ),
    "70b-1layer": ModelConfig(
        num_layers=1,
        num_qo_heads=64,
        num_kv_heads=8,
        hidden_size=8192,
        intermediate_size=28672,
    ),
    "70b": ModelConfig(
        num_layers=80,
        num_qo_heads=64,
        num_kv_heads=8,
        hidden_size=8192,
        intermediate_size=28672,
    ),
}


def bench_one():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=BENCH_FN.keys(), default="punica")
    parser.add_argument("--model", choices=MODEL_CFGS.keys(), default="7b")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--maxlen", type=int, default=2048)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    bench_fn = BENCH_FN[args.system]
    model_cfg = MODEL_CFGS[args.model]
    model_cfg.dtype = args.dtype
    model_cfg.device = args.device

    rs = generate_request_set(
        num_requests=args.batch_size * args.num_batches, maxlen=args.maxlen
    )
    textgen_cfg = TextGenConfig(batch_size=args.batch_size)
    res = bench_fn(model_cfg, textgen_cfg, rs)

    t = res.encode_latency / rs.prompt_lens
    print("num_requests:", len(rs))
    print("batch_size:", textgen_cfg.batch_size)
    print(
        "encode_latency:",
        f"{res.encode_latency.mean()*1e3:.3f}ms ± {res.encode_latency.std()*1e3:.3f}ms per request;",
        f"{t.mean()*1e3:.3f}ms ± {t.std()*1e3:.3f}ms per token",
    )
    t = res.decode_latency / rs.output_lens
    print("decode_latency:", f"{t.mean()*1e3:.3f}ms ± {t.std()*1e3:.3f}ms per token")
    print("total prompt tokens:", rs.prompt_lens.sum())
    print("total new tokens:", rs.output_lens.sum())
    print("duration:", f"{res.duration:.3f}s")
    print(
        "throughput ((prompt+new)/duration):",
        f"{(rs.prompt_lens.sum()+rs.output_lens.sum())/res.duration:.3f} token/s",
    )


if __name__ == "__main__":
    bench_one()
