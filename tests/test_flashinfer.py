import numpy as np
import pytest
import torch

from punica import BatchedKvCache, BatchLenInfo, KvCache, KvPool
from punica.ops import append_kv, batch_decode, batch_prefill, init_kv

num_layers = 3
num_qo_heads = 32
head_dim = 128
batch_size = 7
page_len = 16
maxlen = 500
device = torch.device("cuda:0")


def assert_close(a, b):
    rtol, atol = {
        torch.float16: (1e-3, 5e-4),
        torch.bfloat16: (8e-3, 8e-3),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


def assert_eq(a, b):
    torch.testing.assert_close(a, b, rtol=0, atol=0)


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


def ref_batch_prefill(
    q: torch.Tensor,
    qo_indptr: torch.Tensor,
    cs: list[KvCache],
    layer_idx: int,
) -> torch.Tensor:
    _, n_qo, _ = q.shape
    _l, _2, n_kv, _p, d = cs[0].pool.page_meta.shape
    b = len(cs)
    assert (b + 1,) == qo_indptr.shape
    assert (qo_indptr[-1].item(), n_qo, d) == q.shape

    sm_scale = 1.0 / np.sqrt(d)
    out = []
    for i in range(b):
        assert qo_indptr[i + 1] - qo_indptr[i] == cs[i].seqlen
        s = cs[i].seqlen

        mask = torch.zeros(s, s, dtype=torch.float32, device=q.device)
        mask.masked_fill_(
            torch.ones(s, s, device=q.device, dtype=torch.bool).tril().logical_not(),
            float("-inf"),
        )

        kv_pages = torch.cat(list(cs[i].pages), dim=3)[layer_idx]
        ki = kv_pages[0, :, :s, :].contiguous().to(torch.float32)
        vi = kv_pages[1, :, :s, :].contiguous().to(torch.float32)
        qi = q[qo_indptr[i] : qo_indptr[i + 1]].to(torch.float32)

        qi = rotary_embed(qi.transpose(0, 1), 0).transpose(0, 1)
        ki = rotary_embed(ki, 0)

        ki = repeat_kv(ki, n_qo // n_kv)
        vi = repeat_kv(vi, n_qo // n_kv)

        pi = torch.einsum("qnd,nkd->nqk", qi, ki) * sm_scale
        pi += mask
        pi = torch.softmax(pi, dim=-1)
        oi = torch.einsum("nqs,nsd->qnd", pi, vi).to(q.dtype)
        out.append(oi)
    o = torch.cat(out, dim=0)
    return o


def ref_batch_decode(
    q: torch.Tensor,
    cs: list[KvCache],
    layer_idx: int,
) -> torch.Tensor:
    b, n_qo, _ = q.shape
    _l, _2, n_kv, _p, d = cs[0].pool.page_meta.shape
    assert (b, n_qo, d) == q.shape

    sm_scale = 1.0 / np.sqrt(d)
    out = []
    for i in range(b):
        seqlen = cs[i].seqlen
        kv_pages = torch.cat(list(cs[i].pages), dim=3)[layer_idx]
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
def test_batch_prefill_correctness(dtype_str: str, group_size: int):
    torch.manual_seed(0xABCDABCD987)
    num_kv_heads = num_qo_heads // group_size
    assert num_kv_heads * group_size == num_qo_heads
    dtype = getattr(torch, dtype_str)

    pool = KvPool(num_layers, num_kv_heads, head_dim, page_len, dtype, device)
    seqlens = torch.randint(1, maxlen, (batch_size,), device="cpu").tolist()
    blen = BatchLenInfo(seqlens, 0, device)
    q = torch.randn(sum(seqlens), num_qo_heads, head_dim, dtype=dtype, device=device)
    cs = [KvCache(pool, l) for l in seqlens]
    kv = BatchedKvCache(cs)
    for page in pool.allocated_pages():
        page.copy_(torch.rand_like(page))

    assert blen.indptr is not None
    for layer_idx in range(num_layers):
        o_ref = ref_batch_prefill(q, blen.indptr, cs, layer_idx)
        o_our = batch_prefill(q, blen.indptr, kv, layer_idx)
        assert_close(o_ref, o_our)


@pytest.mark.parametrize("dtype_str", ["float16", "bfloat16"])
@pytest.mark.parametrize("group_size", [1, 8])
@torch.inference_mode()
def test_batch_decode_correctness(dtype_str: str, group_size: int):
    torch.manual_seed(0xABCDABCD987)
    num_kv_heads = num_qo_heads // group_size
    assert num_kv_heads * group_size == num_qo_heads
    dtype = getattr(torch, dtype_str)

    pool = KvPool(num_layers, num_kv_heads, head_dim, page_len, dtype, device)
    seqlens = torch.randint(1, maxlen, (batch_size,), dtype=torch.int32, device="cpu")
    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=dtype, device=device)
    cs = [KvCache(pool, int(l.item())) for l in seqlens]
    kv = BatchedKvCache(cs)
    for page in pool.allocated_pages():
        page.copy_(torch.randn_like(page))

    for layer_idx in range(num_layers):
        o_ref = ref_batch_decode(q, cs, layer_idx)
        o_our = batch_decode(q, kv, layer_idx)
        assert_close(o_ref, o_our)


@pytest.mark.parametrize("dtype_str", ["float16", "bfloat16"])
@pytest.mark.parametrize("group_size", [1, 8])
@torch.inference_mode()
def test_init_kv(dtype_str: str, group_size: int):
    torch.manual_seed(0xABCDABCD987)
    num_kv_heads = num_qo_heads // group_size
    assert num_kv_heads * group_size == num_qo_heads
    dtype = getattr(torch, dtype_str)

    pool = KvPool(num_layers, num_kv_heads, head_dim, page_len, dtype, device)
    seqlens = torch.randint(1, maxlen, (batch_size,), dtype=torch.int32, device="cpu")
    seqlens = seqlens.tolist() + [15, 16, 17, 31, 32, 33]
    total_len = sum(seqlens)
    cs = [KvCache(pool, l) for l in seqlens]
    kv = BatchedKvCache(cs)
    blen = BatchLenInfo(seqlens, 0, device)
    for layer_idx in range(num_layers):
        k = torch.randn(total_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(total_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        assert blen.indptr is not None
        init_kv(kv, k, v, blen.indptr, layer_idx)
        for i, kvcache in enumerate(cs):
            ki = k[blen.indptr[i] : blen.indptr[i + 1]]
            vi = v[blen.indptr[i] : blen.indptr[i + 1]]
            for j, page in enumerate(kvcache.pages):
                s1 = slice(j * page_len, (j + 1) * page_len)
                s2 = (
                    kvcache.pool.page_len
                    if j + 1 < kvcache.num_pages
                    else kv.last_page_offset[i].item()
                )
                assert_eq(ki[s1], page[layer_idx, 0, :, :s2, :].transpose(0, 1))
                assert_eq(vi[s1], page[layer_idx, 1, :, :s2, :].transpose(0, 1))


@pytest.mark.parametrize("dtype_str", ["float16", "bfloat16"])
@pytest.mark.parametrize("group_size", [1, 8])
@torch.inference_mode()
def test_append_kv(dtype_str: str, group_size: int):
    torch.manual_seed(0xABCDABCD987)
    num_kv_heads = num_qo_heads // group_size
    assert num_kv_heads * group_size == num_qo_heads
    dtype = getattr(torch, dtype_str)

    pool = KvPool(num_layers, num_kv_heads, head_dim, page_len, dtype, device)
    seqlens = torch.randint(1, maxlen, (batch_size,), dtype=torch.int32, device="cpu")
    seqlens = seqlens.tolist() + [15, 16, 17, 31, 32, 33]
    bs = len(seqlens)
    cs = [KvCache(pool, l) for l in seqlens]
    kv = BatchedKvCache(cs)
    for layer_idx in range(num_layers):
        k = torch.randn(bs, num_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(bs, num_kv_heads, head_dim, dtype=dtype, device=device)
        append_kv(kv, k, v, layer_idx)
        for i, kvcache in enumerate(cs):
            offset = kv.last_page_offset[i].item() - 1
            assert_eq(k[i], kvcache.pages[-1][layer_idx, 0, :, offset, :])
            assert_eq(v[i], kvcache.pages[-1][layer_idx, 1, :, offset, :])
