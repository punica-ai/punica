# Adapted from HuggingFace Transformers Library
# https://github.com/huggingface/transformers/blob/17a55534f5e5df10ac4804d4270bf6b8cc24998d/src/transformers/models/llama/modeling_llama.py

import math
from typing import Tuple

import torch
from torch import nn
from transformers.models.llama.modeling_llama import (ACT2FN, LlamaConfig,
                                                      LlamaRMSNorm,
                                                      PreTrainedModel,
                                                      rotate_half)

from punica.ops import add_lora, rotary_mha_decode
from punica.utils import CatTensor, KvPool, LlamaLoraModelWeightIndicies


def rotary_pos_emb(q, k, beg):
  device = q.device
  dtype = q.dtype
  bsz, nhead, seqlen, dim = q.shape
  end = beg + seqlen

  base = 10000
  inv_freq = 1.0 / (base**(torch.arange(0, dim, 2).float().to(device) / dim))
  t = torch.arange(beg, end, device=device, dtype=dtype)
  freqs = torch.einsum("i,j->ij", t, inv_freq)
  emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(0).unsqueeze(0)
  cos = emb.cos()
  sin = emb.sin()
  q_embed = (q * cos) + (rotate_half(q) * sin)
  k_embed = (k * cos) + (rotate_half(k) * sin)
  return q_embed.to(q.dtype), k_embed.to(k.dtype)


class LlamaAttentionWithLora(nn.Module):

  def __init__(self, config: LlamaConfig, lora_scale: float, layer_idx: int):
    super().__init__()
    self.config = config
    self.hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.head_dim = self.hidden_size // self.num_heads
    self._scale = 1 / math.sqrt(self.head_dim)
    self.lora_scale = lora_scale
    self.layer_idx = layer_idx

    if (self.head_dim * self.num_heads) != self.hidden_size:
      raise ValueError(
          f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
          f" and `num_heads`: {self.num_heads}).")
    self.q_proj = nn.Linear(
        self.hidden_size, self.num_heads * self.head_dim, bias=False)
    self.k_proj = nn.Linear(
        self.hidden_size, self.num_heads * self.head_dim, bias=False)
    self.v_proj = nn.Linear(
        self.hidden_size, self.num_heads * self.head_dim, bias=False)
    self.o_proj = nn.Linear(
        self.num_heads * self.head_dim, self.hidden_size, bias=False)

  def forward(
      self,
      kvpool: KvPool,
      hidden_states: CatTensor,
      past_lens: torch.LongTensor,
      kvidx: torch.LongTensor,
      lora: LlamaLoraModelWeightIndicies,
  ) -> CatTensor:
    lens = hidden_states.lens

    torch.cuda.nvtx.range_push("qkv_proj")
    q_proj = self.q_proj(hidden_states.cat)
    k_proj = self.k_proj(hidden_states.cat)
    v_proj = self.v_proj(hidden_states.cat)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("lora_qkv")
    add_lora(q_proj, hidden_states.cat, lora.q.mgr.wa_T, lora.q.mgr.wb_T,
             lora.q.indicies, self.layer_idx, self.lora_scale)
    add_lora(k_proj, hidden_states.cat, lora.k.mgr.wa_T, lora.k.mgr.wb_T,
             lora.k.indicies, self.layer_idx, self.lora_scale)
    add_lora(v_proj, hidden_states.cat, lora.v.mgr.wa_T, lora.v.mgr.wb_T,
             lora.v.indicies, self.layer_idx, self.lora_scale)
    torch.cuda.nvtx.range_pop()

    if past_lens[0] == 0:
      q_proj = q_proj.split(lens)
      k_proj = k_proj.split(lens)
      v_proj = v_proj.split(lens)
      kvidx_cpu = kvidx.cpu()
      stack_attn_output = []
      for batch_idx, q_len in enumerate(lens):
        torch.cuda.nvtx.range_push(f"batch_idx={batch_idx}")
        torch.cuda.nvtx.range_push("transpose")
        query_states = q_proj[batch_idx].view(1, q_len, self.num_heads,
                                              self.head_dim).transpose(1, 2)
        key_states = k_proj[batch_idx].view(1, q_len, self.num_heads,
                                            self.head_dim).transpose(1, 2)
        value_states = v_proj[batch_idx].view(1, q_len, self.num_heads,
                                              self.head_dim).transpose(1, 2)
        # (1, n, s, d)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("pos_emb")
        past_len = past_lens[batch_idx]
        query_states, key_states = rotary_pos_emb(query_states, key_states,
                                                  past_len)
        torch.cuda.nvtx.range_pop()

        query_states = query_states.squeeze(0)
        key_states = key_states.squeeze(0)
        value_states = value_states.squeeze(0)
        # (n, s, d)

        torch.cuda.nvtx.range_push("kv_cat")
        past_kv = kvpool.buf[kvidx_cpu[batch_idx].item()]
        past_kv[self.layer_idx, 0,
                past_len:past_len + q_len] = key_states.transpose(0, 1)
        past_kv[self.layer_idx, 1,
                past_len:past_len + q_len] = value_states.transpose(0, 1)
        key_states = past_kv[self.layer_idx,
                             0, :past_len + q_len].transpose(0, 1)
        value_states = past_kv[self.layer_idx,
                               1, :past_len + q_len].transpose(0, 1)
        torch.cuda.nvtx.range_pop()

        # scaled dot product attention
        torch.cuda.nvtx.range_push("sdpa")
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, is_causal=True)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(q_len, self.hidden_size)
        stack_attn_output.append(attn_output)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
      attn_outputs = torch.cat(stack_attn_output, dim=0)
    else:
      torch.cuda.nvtx.range_push(f"batched_flash_attn")
      q = q_proj.view(len(lens), self.num_heads, self.head_dim)
      k = k_proj.view(len(lens), self.num_heads, self.head_dim)
      v = v_proj.view(len(lens), self.num_heads, self.head_dim)
      attn_outputs = rotary_mha_decode(q, k, v, past_lens, kvpool.buf, kvidx,
                                       self.layer_idx)
      attn_outputs = attn_outputs.view(len(lens), self.hidden_size)
      torch.cuda.nvtx.range_pop()

    # output projection
    torch.cuda.nvtx.range_push("o_proj")
    o_proj = self.o_proj(attn_outputs)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("lora_o")
    add_lora(o_proj, attn_outputs, lora.o.mgr.wa_T, lora.o.mgr.wb_T,
             lora.o.indicies, self.layer_idx, self.lora_scale)
    torch.cuda.nvtx.range_pop()

    o_proj = CatTensor(o_proj, lens)
    return o_proj


class LlamaMlpWithLora(nn.Module):

  def __init__(self, config: LlamaConfig, lora_scale: float, layer_idx: int):
    super().__init__()
    self.lora_scale = lora_scale
    self.layer_idx = layer_idx
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
    self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
    self.act_fn = ACT2FN[config.hidden_act]

  def forward(
      self,
      x: torch.Tensor,
      lora: LlamaLoraModelWeightIndicies,
  ) -> torch.Tensor:
    torch.cuda.nvtx.range_push("gate_proj")
    gate = self.gate_proj(x)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("lora_gate")
    add_lora(gate, x, lora.gate.mgr.wa_T, lora.gate.mgr.wb_T,
             lora.gate.indicies, self.layer_idx, self.lora_scale)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("gate_act")
    gate = self.act_fn(gate)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("up_proj")
    up = self.up_proj(x)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("lora_up")
    add_lora(up, x, lora.up.mgr.wa_T, lora.up.mgr.wb_T, lora.up.indicies,
             self.layer_idx, self.lora_scale)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("gate_up")
    t = gate * up
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("down_proj")
    down = self.down_proj(t)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("lora_down")
    add_lora(down, t, lora.down.mgr.wa_T, lora.down.mgr.wb_T,
             lora.down.indicies, self.layer_idx, self.lora_scale)
    torch.cuda.nvtx.range_pop()
    return down


class LlamaDecoderLayerWithLora(nn.Module):

  def __init__(self, config: LlamaConfig, lora_scale: float, layer_idx: int):
    super().__init__()
    self.hidden_size = config.hidden_size
    self.self_attn = LlamaAttentionWithLora(config, lora_scale, layer_idx)
    self.mlp = LlamaMlpWithLora(config, lora_scale, layer_idx)
    self.input_layernorm = LlamaRMSNorm(
        config.hidden_size, eps=config.rms_norm_eps)
    self.post_attention_layernorm = LlamaRMSNorm(
        config.hidden_size, eps=config.rms_norm_eps)

  def forward(
      self,
      kvpool: KvPool,
      hidden_states: CatTensor,
      past_lens: torch.LongTensor,
      kvidx: torch.LongTensor,
      lora: LlamaLoraModelWeightIndicies,
  ) -> torch.Tensor:
    lens = hidden_states.lens

    hidden_states = hidden_states.cat
    residual = hidden_states

    torch.cuda.nvtx.range_push("input_norm")
    hidden_states = self.input_layernorm(hidden_states)
    torch.cuda.nvtx.range_pop()

    # Self Attention
    torch.cuda.nvtx.range_push("LlamaAttention")
    hidden_states = self.self_attn(kvpool, CatTensor(hidden_states, lens),
                                   past_lens, kvidx, lora)
    hidden_states = hidden_states.cat
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
    hidden_states = self.mlp(hidden_states, lora)
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("r")
    hidden_states = residual + hidden_states
    torch.cuda.nvtx.range_pop()

    hidden_states = CatTensor(hidden_states, lens)
    return hidden_states


class LlamaPreTrainedModel(PreTrainedModel):
  config_class = LlamaConfig
  base_model_prefix = "model"
  supports_gradient_checkpointing = False
  _no_split_modules = ["LlamaDecoderLayerWithLora"]
  _keys_to_ignore_on_load_unexpected = [
      r"decoder\.version",
      r"self_attn\.rotary_emb\.inv_freq",
  ]


class LlamaModelWithLora(LlamaPreTrainedModel):

  def __init__(self, config: LlamaConfig, lora_scale: float):
    super().__init__(config)
    self.padding_idx = config.pad_token_id
    self.vocab_size = config.vocab_size
    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                     self.padding_idx)
    self.layers = nn.ModuleList([
        LlamaDecoderLayerWithLora(config, lora_scale, layer_idx=i)
        for i in range(config.num_hidden_layers)
    ])
    self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.post_init()

  def forward(
      self,
      kvpool: KvPool,
      input_ids: CatTensor,
      past_lens: torch.LongTensor,
      kvidx: torch.LongTensor,
      lora: LlamaLoraModelWeightIndicies,
  ) -> CatTensor:
    torch.cuda.nvtx.range_push(f"embed")
    hidden_states = CatTensor(self.embed_tokens(input_ids.cat), input_ids.lens)
    torch.cuda.nvtx.range_pop()

    for layer_idx, decoder_layer in enumerate(self.layers):
      torch.cuda.nvtx.range_push(f"layer={layer_idx}")
      hidden_states = decoder_layer(kvpool, hidden_states, past_lens, kvidx,
                                    lora)
      torch.cuda.nvtx.range_pop()

    lens = hidden_states.lens
    torch.cuda.nvtx.range_push("lastnorm")
    hidden_states = self.norm(hidden_states.cat)
    torch.cuda.nvtx.range_pop()
    hidden_states = CatTensor(hidden_states, lens)

    return hidden_states


class LlamaForCausalLMWithLora(LlamaPreTrainedModel):

  def __init__(self, config: LlamaConfig, lora_scale: float):
    super().__init__(config)
    self.model = LlamaModelWithLora(config, lora_scale)
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    self.post_init()

  def forward(
      self,
      kvpool: KvPool,
      input_ids: CatTensor,
      past_lens: torch.LongTensor,
      kvidx: torch.LongTensor,
      lora: LlamaLoraModelWeightIndicies,
  ) -> Tuple[CatTensor, CatTensor]:
    torch.cuda.nvtx.range_push("LlamaForCausalLMWithLora")
    hidden_states = self.model(kvpool, input_ids, past_lens, kvidx, lora)
    lens = hidden_states.lens
    torch.cuda.nvtx.range_push("lm_head")
    logits = self.lm_head(hidden_states.cat)
    torch.cuda.nvtx.range_pop()
    logits = CatTensor(logits, lens)
    torch.cuda.nvtx.range_pop()
    return logits, hidden_states
