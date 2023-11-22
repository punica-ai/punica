from typing import Sequence

import torch


class KvPool:

  def __init__(
      self,
      num_layers: int,
      num_heads: int,
      head_dim: int,
      capacity: int,
      block_len: int,
      dtype: torch.dtype,
      device: torch.device,
  ):
    self._buf = torch.empty(
        (capacity, num_layers, 2, num_heads, block_len, head_dim),
        dtype=dtype,
        device=device)
    self._free = set(range(capacity))

  @property
  def buf(self):
    return self._buf

  @property
  def num_layers(self):
    c, l, _, n, p, d = self._buf.shape
    return l

  @property
  def block_len(self):
    c, l, _, n, p, d = self._buf.shape
    return p

  @property
  def num_free_blocks(self):
    return len(self._free)

  def alloc_block(self) -> int:
    idx = self._free.pop()
    return idx

  def free_block(self, idx: int):
    assert 0 <= idx < self._buf.size(0)
    assert idx not in self._free
    self._free.add(idx)


class KvCache:
  """Key-value cache for one sequence."""

  def __init__(self, pool: KvPool, init_len: int):
    if init_len < 0:
      raise ValueError("init_len must be non-negative")

    self._pool = pool
    if init_len > 0:
      blocks = (init_len + pool.block_len - 1) // pool.block_len
      self._indicies = [pool.alloc_block() for _ in range(blocks)]
      self._seqlen = init_len
    else:
      self._indicies = []
      self._seqlen = 0

  @property
  def pool(self) -> KvPool:
    return self._pool

  @property
  def seqlen(self) -> int:
    return self._seqlen

  @property
  def indicies(self) -> list[int]:
    return self._indicies

  def acquire_one(self):
    """Reserve space for one more token"""
    last_page_offset = (self._seqlen - 1) % self._pool.block_len + 1
    if last_page_offset == self._pool.block_len:
      self._indicies.append(self._pool.alloc_block())
    self._seqlen += 1

  def release(self):
    """Release all blocks"""
    self._seqlen = 0
    for idx in self._indicies:
      self._pool.free_block(idx)
    self._indicies.clear()


class BatchedKvCache:
  """Key-value cache for a batch of sequences."""

  def __init__(self, kv: Sequence[KvCache]):
    assert len(kv) > 0
    pool = kv[0].pool
    device = pool.buf.device
    indptr = [0]
    indicies = []
    last_page_offset = []
    for c in kv:
      assert c.pool is pool
      indptr.append(indptr[-1] + len(c.indicies))
      indicies.extend(c.indicies)
      last_page_offset.append((c.seqlen - 1) % pool.block_len + 1)

    self.data = pool.buf
    self.indptr = torch.tensor(indptr, dtype=torch.int32, device=device)
    self.indicies = torch.tensor(indicies, dtype=torch.int32, device=device)
    self.last_page_offset = torch.tensor(
        last_page_offset, dtype=torch.int32, device=device)

  @property
  def page_size(self):
    return self.data.size(-2)
