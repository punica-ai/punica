import numpy as np
import pytest
import torch

from punica.utils.kvcache import BatchedKvCache, GrowableTensor, KvCache, KvPool

num_layers = 24
num_heads = 32
head_dim = 128
page_len = 16
dtype = torch.float16
device = torch.device("cuda:0")


def test_KvPool():
    pool = KvPool(num_layers, num_heads, head_dim, page_len, dtype, device)
    shape = (num_layers, 2, num_heads, page_len, head_dim)
    assert pool.num_layers == num_layers
    assert pool.num_heads == num_heads
    assert pool.head_dim == head_dim
    assert pool.page_len == page_len
    assert pool.dtype == dtype
    assert pool.device == device
    assert pool.page_meta.shape == shape
    assert pool.page_nbytes == np.prod(shape) * 2

    pages = []
    assert pool.num_pages == 0
    for _ in range(10):
        page = pool.alloc_page()
        assert page.shape == (num_layers, 2, num_heads, page_len, head_dim)
        assert page.dtype == dtype
        assert page.device == device
        pages.append(page)
        assert pool.num_pages == len(pages)

    while pages:
        page = pages.pop()
        pool.free_page(page)
        assert pool.num_pages == len(pages)


def test_GrowableTensor_empty():
    t = GrowableTensor([], dtype, device)
    assert t.view().shape == (0,)
    assert t.view().dtype == dtype
    assert t.view().device == device


def test_GrowableTensor_grow_from_empty():
    t = GrowableTensor([], dtype, device)
    for i in range(10):
        t.append(i)
        assert t.view().shape == (i + 1,)
        assert t.view().dtype == dtype
        assert t.view().device == device
        assert t.view().tolist() == list(range(i + 1))


def test_GrowableTensor_grow_from_data():
    data = [1, 2, 3, 4, 5]
    t = GrowableTensor(data, dtype, device)
    for i in range(10):
        t.append(i)
        assert t.view().shape == (len(data) + i + 1,)
        assert t.view().dtype == dtype
        assert t.view().device == device
        assert t.view().tolist() == data + list(range(i + 1))


def test_GrowableTensor_clear():
    data = [1, 2, 3, 4, 5]
    t = GrowableTensor(data, dtype, device)
    t.clear()
    assert t.view().shape == (0,)
    assert t.view().dtype == dtype
    assert t.view().device == device
    assert t.view().tolist() == []
    for i in range(10):
        t.append(i)
        assert t.view().shape == (i + 1,)
        assert t.view().dtype == dtype
        assert t.view().device == device
        assert t.view().tolist() == list(range(i + 1))


@pytest.fixture
def pool():
    return KvPool(num_layers, num_heads, head_dim, page_len, dtype, device)


@pytest.mark.parametrize("init_len", [0, 1, 15, 16, 17, 31, 32, 33])
def test_KvCache(pool: KvPool, init_len: int):
    kvcache = KvCache(pool, init_len)
    assert kvcache.pool is pool
    assert kvcache.seqlen == init_len
    assert kvcache.num_pages == (init_len + page_len - 1) // page_len
    assert kvcache.ptrs.shape == (kvcache.num_pages,)
    assert kvcache.ptrs.dtype == torch.int64
    assert kvcache.ptrs.device == device
    assert pool.num_pages == kvcache.num_pages

    for i in range(1, 65):
        kvcache.acquire_one()
        assert kvcache.seqlen == init_len + i
        assert kvcache.num_pages == (kvcache.seqlen + page_len - 1) // page_len
        assert kvcache.ptrs.shape == (kvcache.num_pages,)
        assert kvcache.ptrs.dtype == torch.int64
        assert kvcache.ptrs.device == device
        assert pool.num_pages == kvcache.num_pages

    kvcache.release()
    assert kvcache.seqlen == 0
    assert kvcache.num_pages == 0
    assert kvcache.ptrs.shape == (0,)
    assert kvcache.ptrs.dtype == torch.int64
    assert kvcache.ptrs.device == device
    assert pool.num_pages == 0


def test_BatchedKvCache(pool: KvPool):
    seqlens = [15, 16, 17, 31, 32, 33]
    num_pages = [1, 1, 2, 2, 2, 3]
    kv_list = [KvCache(pool, seqlen) for seqlen in seqlens]
    assert num_pages == [kv.num_pages for kv in kv_list]
    assert pool.num_pages == sum(num_pages)
    batched_kv = BatchedKvCache(kv_list)
    assert batched_kv.pool is pool
    assert batched_kv.ptrs.shape == (sum(num_pages),)
    assert batched_kv.ptrs.dtype == torch.int64
    assert batched_kv.ptrs.device == device
    assert batched_kv.indptr.tolist() == [0, 1, 2, 4, 6, 8, 11]
    assert batched_kv.last_page_offset.tolist() == [15, 16, 1, 15, 16, 1]
