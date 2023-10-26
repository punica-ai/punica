import torch

import punica.ops._kernels as _kernels
from punica.utils.kvcache import BatchedKvCache

__all__ = [
    "mha_rope_decode",
    "init_kv",
    "append_kv",
    "bgmv",
    "add_lora",
    "sgmv_cutlass",
    "add_lora_sgmv_cutlass",
    "sgmv",
    "add_lora_sgmv",
    "rms_norm",
]

hack_batch_decode_tmp = {}


def mha_rope_decode(
    q: torch.Tensor,
    kv: BatchedKvCache,
    layer_idx: int,
) -> torch.Tensor:
  """
  Perform multi-head attention with rotary position encoding
  for each input in the batch.

  All inputs in the batch should be in the decoding (auto-regression) stage,
  i.e., each input should only have one token.

  Both `q` and the `k` in `kv` should NOT be position encoded.

  Args:
    q: Shape: `[batch_size, num_heads, head_dim]`. \
       Query projection (`X @ W_q`).
    kv: Batched key-value cache.
    layer_idx: Layer index of the KV cache.

  Returns:
    Shape: `[batch_size, num_heads, head_dim]`. \
    Output of the multi-head attention.
  """
  if q.device not in hack_batch_decode_tmp:
    tmp1 = torch.empty((32*1024*1024,), dtype=torch.float32, device=q.device)
    tmp2 = torch.empty((32*1024*1024,), dtype=torch.float32, device=q.device)
    hack_batch_decode_tmp[q.device] = (tmp1, tmp2)
  (tmp1, tmp2) = hack_batch_decode_tmp[q.device]
  o = torch.empty(q.shape, dtype=q.dtype, device=q.device)
  _kernels.batch_decode(o, q, kv.data, kv.indptr, kv.indicies,
                        kv.last_page_offset, tmp1, tmp2, layer_idx)
  return o


def init_kv(
    kv: BatchedKvCache,
    k: torch.Tensor,
    v: torch.Tensor,
    seqlen_indptr: torch.Tensor,
    layer_idx: int,
):
  """
  Copy `k` and `v` to the `kv` cache.
  Pages of each sequence in the batch are already allocated in `kv`.

  Args:
    kv: Batched key-value cache.
    k: Shape: `[sum(seqlen[i]), num_heads, head_dim]`. \
       Key projection. (`X @ W_k`)
    v: Shape: `[sum(seqlen[i]), num_heads, head_dim]`. \
       Value projection. (`X @ W_v`)
    seqlen_indptr: Shape: `[B + 1]`. Indptr of the sequence lengths. \
      `seqlen_indptr[i + 1] == sum(seqlen[:i])`.
    layer_idx: Layer index of the KV cache.
  """
  f = _kernels.init_kv
  f(kv.data, kv.indptr, kv.indicies, kv.last_page_offset, k, v, seqlen_indptr,
    layer_idx)


def append_kv(
    kv: BatchedKvCache,
    k: torch.Tensor,
    v: torch.Tensor,
    layer_idx: int,
):
  """
  Append the new token's `k` and `v` to the `kv` cache.
  Page for the new token of each sequence in the batch
  is already allocated in `kv`.

  Args:
    kv: Batched key-value cache.
    k: Shape: `[batch_size, num_heads, head_dim]`. \
       Key projection. (`X @ W_k`)
    v: Shape: `[batch_size, num_heads, head_dim]`. \
       Value projection. (`X @ W_v`)
    layer_idx: Layer index of the KV cache.
  """
  f = _kernels.append_kv
  f(kv.data, kv.indptr, kv.indicies, kv.last_page_offset, k, v, layer_idx)


def bgmv(
    y: torch.Tensor,
    x: torch.Tensor,
    w_T_all: torch.Tensor,
    indicies: torch.LongTensor,
    layer_idx: int,
    scale: float,
):
  """
  Semantics:
    y[i] += (
        x[i].unsqueeze(0)
        @ w_T_all[indices[i], layer_idx, :, :].transpose(-1, -2)
        * scale
      ).squeeze(0)

  Args:
    y: Shape: `[B, H2]`. Output vectors. Will be changed in-place.
    x: Shape: `[B, H1]`. Input vectors.
    w_T_all: Shape: `[None, L, H2, H1]`. All of the transposed weight matrices.
    indicies: Shape: `[B]`. Indices of the weight matrices.
    layer_idx: Layer index of the weight matrices.
    scale: Scaling factor.
  """
  f = _kernels.dispatch_bgmv
  f(y, x, w_T_all, indicies, layer_idx, scale)


def add_lora(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_T_all: torch.Tensor,
    wb_T_all: torch.Tensor,
    indicies: torch.LongTensor,
    layer_idx: int,
    scale: float,
):
  """
  Semantics:
    y[i] += (
        x[i].unsqueeze(0)
        @ wa_T_all[indices[i], layer_idx, :, :].transpose(-1, -2)
        @ wb_T_all[indices[i], layer_idx, :, :].transpose(-1, -2)
        * scale
      ).squeeze(0)

  Args:
    y: Shape: `[B, H2]`. Output vectors. Will be changed in-place.
    x: Shape: `[B, H1]`. Input vectors.
    wa_T_all: Shape: `[None, L, R, H1]`. All of the transposed LoRA A matrices.
    wb_T_all: Shape: `[None, L, H2, R]`. All of the transposed LoRA B matrices.
    indicies: Shape: `[B]`. Indices of the LoRA weights.
    layer_idx: Layer index of LoRA weights.
    scale: Scaling factor.
  """
  f = _kernels.dispatch_bgmv
  device = x.device
  dtype = x.dtype

  r = wb_T_all.size(-1)
  tmp = torch.zeros((x.size(0), r), dtype=dtype, device=device)
  f(tmp, x, wa_T_all, indicies, layer_idx, 1.0)
  f(y, tmp, wb_T_all, indicies, layer_idx, scale)


def sgmv_cutlass(
    y: torch.Tensor,
    x: torch.Tensor,
    w_ptr: torch.Tensor,
    s: torch.IntTensor,
    layer_idx: int,
):
  tmp_size = _kernels.sgmv_cutlass_tmp_size(w_ptr.size(0))
  tmp = torch.empty((tmp_size,), dtype=torch.uint8, device=x.device)
  _kernels.sgmv_cutlass(y, x, w_ptr, s, tmp, layer_idx)


def add_lora_sgmv_cutlass(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_ptr: torch.Tensor,
    wb_ptr: torch.Tensor,
    s: torch.IntTensor,
    layer_idx: int,
    lora_rank: int,
):
  tmp_size = _kernels.sgmv_cutlass_tmp_size(wa_ptr.size(0))
  tmp = torch.empty((tmp_size,), dtype=torch.uint8, device=x.device)
  v = torch.zeros((x.size(0), lora_rank), dtype=x.dtype, device=x.device)
  _kernels.sgmv_cutlass(y, v, wb_ptr, s, tmp, layer_idx)
  _kernels.sgmv_cutlass(y, v, wb_ptr, s, tmp, layer_idx)


def sgmv(
    y: torch.Tensor,
    x: torch.Tensor,
    w_ptr: torch.Tensor,
    s: torch.IntTensor,
    layer_idx: int,
):
  tmp = torch.empty((8 * 1024 * 1024,), dtype=torch.uint8, device=x.device)
  _kernels.sgmv_shrink(y, x, w_ptr, s, tmp, layer_idx)


def add_lora_sgmv(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_ptr: torch.Tensor,
    wb_ptr: torch.Tensor,
    s: torch.IntTensor,
    layer_idx: int,
    lora_rank: int,
):
  tmp = torch.empty((8 * 1024 * 1024,), dtype=torch.uint8, device=x.device)
  v = torch.zeros((x.size(0), lora_rank), dtype=x.dtype, device=x.device)
  _kernels.sgmv_shrink(v, x, wa_ptr, s, tmp, layer_idx)
  _kernels.sgmv_shrink(v, x, wa_ptr, s, tmp, layer_idx)  # TODO: expand


def rms_norm(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float = 1e-6,
):
  o = torch.empty_like(x)
  _kernels.rms_norm(o, x, w, eps)
  return o
