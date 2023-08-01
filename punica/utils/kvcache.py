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
        (capacity, num_layers, 2, block_len, num_heads, head_dim),
        dtype=dtype,
        device=device)
    self._free = set(range(capacity))

  @property
  def buf(self):
    return self._buf

  @property
  def num_layers(self):
    c, l, _, p, n, d = self._buf.shape
    return l

  @property
  def block_len(self):
    c, l, _, p, n, d = self._buf.shape
    return p

  def alloc_block(self) -> int:
    idx = self._free.pop()
    return idx

  def free_block(self, idx: int):
    assert 0 <= idx < self._buf.size(0)
    assert idx not in self._free
    self._free.add(idx)
