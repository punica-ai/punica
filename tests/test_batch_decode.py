import numpy as np
import pytest
import torch

from punica import BatchedKvCache, KvCache, KvPool
from punica.ops import batch_decode


def assert_close(a, b):
    rtol, atol = {
        torch.float16: (5e-4, 5e-4),
        torch.bfloat16: (1e-3, 2e-3),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rotary_embed(q, beg):
    device = q.device
    dtype = q.dtype
    dim = q.size(-1)
    l = q.size(-2) if q.dim() == 3 else 1

    base = 1e4
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    t = torch.arange(beg, beg + l, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed


def repeat_kv(t: torch.Tensor, repeat: int) -> torch.Tensor:
    if repeat == 1:
        return t
    num_kv_heads, seqlen, head_dim = t.shape
    t = t[:, None, :, :].expand(num_kv_heads, repeat, seqlen, head_dim)
    t = t.reshape(num_kv_heads * repeat, seqlen, head_dim)
    return t


def ref_batch_decode(
    q: torch.Tensor,
    kv: BatchedKvCache,
    layer_idx: int,
) -> torch.Tensor:
    b, n_qo, _ = q.shape
    _c, _l, _2, n_kv, p, d = kv.data.shape
    assert (b, n_qo, d) == q.shape

    sm_scale = 1.0 / np.sqrt(d)
    out = []
    for i in range(b):
        seqlen = (kv.indptr[i + 1] - kv.indptr[i] - 1) * p + kv.last_page_offset[i]
        kv_pages = torch.cat(
            [
                kv.data[page_idx, layer_idx]
                for page_idx in kv.indicies[kv.indptr[i] : kv.indptr[i + 1]]
            ],
            dim=2,
        )
        ki = kv_pages[0, :, :seqlen, :].contiguous().to(torch.float32)
        vi = kv_pages[1, :, :seqlen, :].contiguous().to(torch.float32)
        qi = q[i].to(torch.float32)

        qi = rotary_embed(qi, seqlen - 1)
        ki = rotary_embed(ki, 0)

        ki = repeat_kv(ki, n_qo // n_kv)
        vi = repeat_kv(vi, n_qo // n_kv)

        pi = torch.einsum("nd,nsd->ns", qi, ki) * sm_scale
        pi = torch.softmax(pi, dim=-1)
        oi = torch.einsum("ns,nsd->nd", pi, vi).to(q.dtype)
        out.append(oi)
    o = torch.stack(out)
    return o


@pytest.mark.parametrize("dtype_str", ["float16", "bfloat16"])
@pytest.mark.parametrize("group_size", [1, 8])
@torch.inference_mode()
def test_batch_decode_correctness(dtype_str: str, group_size: int):
    torch.manual_seed(0xABCDABCD987)
    num_layers = 3
    num_qo_heads = 32
    num_kv_heads = num_qo_heads // group_size
    assert num_kv_heads * group_size == num_qo_heads
    head_dim = 128
    batch_size = 7
    block_len = 16
    maxlen = 500
    dtype = getattr(torch, dtype_str)
    device = torch.device("cuda:0")

    pool = KvPool(
        num_layers=num_layers,
        num_heads=num_kv_heads,
        head_dim=head_dim,
        capacity=(maxlen + block_len - 1) // block_len * batch_size,
        block_len=block_len,
        dtype=dtype,
        device=device,
    )

    seqlens = torch.randint(1, maxlen, (batch_size,), dtype=torch.int32, device="cpu")
    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=dtype, device=device)
    pool.buf.copy_(torch.randn_like(pool.buf))

    cs = [KvCache(pool, int(l.item())) for l in seqlens]
    kv = BatchedKvCache(cs)

    for layer_idx in range(num_layers):
        o_ref = ref_batch_decode(q, kv, layer_idx)
        o_our = batch_decode(q, kv, layer_idx)
        assert_close(o_ref, o_our)
