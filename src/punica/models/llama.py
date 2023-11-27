# Adapted from HuggingFace Transformers Library
# https://github.com/huggingface/transformers/blob/17a55534f5e5df10ac4804d4270bf6b8cc24998d/src/transformers/models/llama/modeling_llama.py

import math

import torch
from torch import nn
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaMLP,
    PreTrainedModel,
)

from punica.ops import append_kv, batch_decode, batch_prefill, init_kv, rms_norm
from punica.utils import BatchedKvCache, BatchLenInfo


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_qo_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_qo_heads // self.num_kv_heads
        self.head_dim = self.hidden_size // self.num_qo_heads
        self._scale = 1 / math.sqrt(self.head_dim)
        self.layer_idx = layer_idx

        assert self.head_dim * self.num_qo_heads == self.hidden_size
        assert self.num_kv_heads * self.num_kv_groups == self.num_qo_heads
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_qo_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_qo_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        blen: BatchLenInfo,
        prefill_kv: BatchedKvCache | None,
        decode_kv: BatchedKvCache | None,
    ) -> torch.Tensor:
        torch.cuda.nvtx.range_push("qkv_proj")
        q_proj = self.q_proj(hidden_states)
        k_proj = self.k_proj(hidden_states)
        v_proj = self.v_proj(hidden_states)
        torch.cuda.nvtx.range_pop()
        stack_attn_output = []

        if len(blen.prefills) > 0:
            assert prefill_kv is not None
            assert blen.indptr is not None
            q = q_proj[: blen.doff].view(blen.doff, self.num_qo_heads, self.head_dim)
            k = k_proj[: blen.doff].view(blen.doff, self.num_kv_heads, self.head_dim)
            v = v_proj[: blen.doff].view(blen.doff, self.num_kv_heads, self.head_dim)

            torch.cuda.nvtx.range_push("init_kv")
            init_kv(prefill_kv, k, v, blen.indptr, self.layer_idx)
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("batch_prefill")
            attn_output = batch_prefill(q, blen.indptr, prefill_kv, self.layer_idx)
            attn_output = attn_output.view(blen.doff, self.hidden_size)
            stack_attn_output.append(attn_output)
            torch.cuda.nvtx.range_pop()

        if blen.decode > 0:
            q = q_proj[blen.doff :].view(blen.decode, self.num_qo_heads, self.head_dim)
            k = k_proj[blen.doff :].view(blen.decode, self.num_kv_heads, self.head_dim)
            v = v_proj[blen.doff :].view(blen.decode, self.num_kv_heads, self.head_dim)

            torch.cuda.nvtx.range_push("append_kv")
            assert decode_kv is not None
            append_kv(decode_kv, k, v, self.layer_idx)
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("batch_decode")
            attn_outputs = batch_decode(q, decode_kv, self.layer_idx)
            attn_outputs = attn_outputs.view(blen.decode, self.hidden_size)
            stack_attn_output.append(attn_outputs)
            torch.cuda.nvtx.range_pop()

        if len(stack_attn_output) == 1:
            attn_outputs = stack_attn_output[0]
        else:
            attn_outputs = torch.cat(stack_attn_output, dim=0)

        # output projection
        torch.cuda.nvtx.range_push("o_proj")
        attn_output = self.o_proj(attn_outputs)
        torch.cuda.nvtx.range_pop()

        return attn_output


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return rms_norm(hidden_states, self.weight, self.variance_epsilon)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        blen: BatchLenInfo,
        prefill_kv: BatchedKvCache | None,
        decode_kv: BatchedKvCache | None,
    ) -> torch.Tensor:
        residual = hidden_states

        torch.cuda.nvtx.range_push("input_norm")
        hidden_states = self.input_layernorm(hidden_states)
        torch.cuda.nvtx.range_pop()

        # Self Attention
        torch.cuda.nvtx.range_push("LlamaAttention")
        hidden_states = self.self_attn(hidden_states, blen, prefill_kv, decode_kv)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("r")
        hidden_states = residual + hidden_states
        torch.cuda.nvtx.range_pop()

        # Fully Connected
        residual = hidden_states
        torch.cuda.nvtx.range_push("norm")
        hidden_states = self.post_attention_layernorm(hidden_states)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("mlp")
        hidden_states = self.mlp(hidden_states)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("r")
        hidden_states = residual + hidden_states
        torch.cuda.nvtx.range_pop()

        return hidden_states


class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["LlamaDecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.version",
        r"self_attn\.rotary_emb\.inv_freq",
    ]


class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        blen: BatchLenInfo,
        prefill_kv: BatchedKvCache | None,
        decode_kv: BatchedKvCache | None,
    ) -> torch.Tensor:
        torch.cuda.nvtx.range_push("embed")
        hidden_states = self.embed_tokens(input_ids)
        torch.cuda.nvtx.range_pop()

        for layer_idx, decoder_layer in enumerate(self.layers):
            torch.cuda.nvtx.range_push(f"layer={layer_idx}")
            hidden_states = decoder_layer(hidden_states, blen, prefill_kv, decode_kv)
            torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("lastnorm")
        hidden_states = self.norm(hidden_states)
        torch.cuda.nvtx.range_pop()

        return hidden_states


class LlamaForCausalLM(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        blen: BatchLenInfo,
        prefill_kv: BatchedKvCache | None,
        decode_kv: BatchedKvCache | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        torch.cuda.nvtx.range_push("LlamaForCausalLM")
        hidden_states = self.model(input_ids, blen, prefill_kv, decode_kv)
        torch.cuda.nvtx.range_push("lm_head")
        logits = self.lm_head(hidden_states)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
        return logits, hidden_states
