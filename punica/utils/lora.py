import dataclasses
from typing import Sequence

import torch

__all__ = [
    "LoraManager",
    "LoraWeight",
    "LoraWeightIndices",
    "LlamaLoraModelWeight",
    "LlamaLoraModelWeightIndicies",
    "LlamaLoraManager",
]


class LoraManager:

  def __init__(
      self,
      capacity: int,
      num_layers: int,
      in_features: int,
      out_features: int,
      lora_rank: int,
      dtype: torch.dtype,
      device: torch.device,
  ):
    self._wa_T = torch.empty((capacity, num_layers, lora_rank, in_features),
                             dtype=dtype,
                             device=device)
    self._wb_T = torch.empty((capacity, num_layers, out_features, lora_rank),
                             dtype=dtype,
                             device=device)
    self._free = set(range(capacity))

  @property
  def device(self) -> torch.device:
    return self._wa_T.device

  @property
  def dtype(self) -> torch.dtype:
    return self._wa_T.dtype

  @property
  def capacity(self) -> int:
    return self._wa_T.size(0)

  @property
  def num_layers(self) -> int:
    return self._wa_T.size(1)

  @property
  def in_features(self) -> int:
    return self._wa_T.size(3)

  @property
  def out_features(self) -> int:
    return self._wb_T.size(2)

  @property
  def lora_rank(self) -> int:
    return self._wa_T.size(2)

  @property
  def wa_T(self) -> torch.Tensor:
    return self._wa_T

  @property
  def wb_T(self) -> torch.Tensor:
    return self._wb_T

  def alloc(self) -> "LoraWeight":
    idx = self._free.pop()
    return LoraWeight(self, idx)

  def free(self, lora_weight: "LoraWeight"):
    assert lora_weight.mgr is self
    idx = lora_weight.idx
    assert 0 <= idx < self.capacity
    assert idx not in self._free
    self._free.add(idx)


@dataclasses.dataclass
class LoraWeight:
  mgr: LoraManager
  idx: int


@dataclasses.dataclass
class LoraWeightIndices:
  mgr: LoraManager
  indicies: torch.LongTensor


@dataclasses.dataclass
class LlamaLoraModelWeight:
  q: LoraWeight
  k: LoraWeight
  v: LoraWeight
  o: LoraWeight
  gate: LoraWeight
  up: LoraWeight
  down: LoraWeight


class LlamaLoraModelWeightIndicies:
  q: LoraWeightIndices
  k: LoraWeightIndices
  v: LoraWeightIndices
  o: LoraWeightIndices
  gate: LoraWeightIndices
  up: LoraWeightIndices
  down: LoraWeightIndices

  def __init__(self, model_weights: Sequence[LlamaLoraModelWeight]):
    device = model_weights[0].q.mgr.device
    for f in ["q", "k", "v", "o", "gate", "up", "down"]:
      mgr = getattr(model_weights[0], f).mgr
      assert mgr.device == device
      idx = [getattr(w, f).idx for w in model_weights]
      assert all(getattr(w, f).mgr is mgr for w in model_weights)
      indicies = torch.tensor(idx, dtype=torch.long, device=device)
      setattr(self, f, LoraWeightIndices(mgr, indicies))


class LlamaLoraManager:

  def __init__(
      self,
      capacity: int,
      num_layers: int,
      hidden_size: int,
      intermediate_size: int,
      lora_rank: int,
      dtype: torch.dtype,
      device: torch.device,
  ):
    self.mgr_hh = LoraManager(capacity * 4, num_layers, hidden_size,
                              hidden_size, lora_rank, dtype, device)
    self.mgr_hi = LoraManager(capacity * 2, num_layers, hidden_size,
                              intermediate_size, lora_rank, dtype, device)
    self.mgr_ih = LoraManager(capacity, num_layers, intermediate_size,
                              hidden_size, lora_rank, dtype, device)

  def alloc(self) -> LlamaLoraModelWeight:
    return LlamaLoraModelWeight(
        q=self.mgr_hh.alloc(),
        k=self.mgr_hh.alloc(),
        v=self.mgr_hh.alloc(),
        o=self.mgr_hh.alloc(),
        gate=self.mgr_hi.alloc(),
        up=self.mgr_hi.alloc(),
        down=self.mgr_ih.alloc(),
    )

  def free(self, weight: LlamaLoraModelWeight):
    self.mgr_hh.free(weight.q)
    self.mgr_hh.free(weight.k)
    self.mgr_hh.free(weight.v)
    self.mgr_hh.free(weight.o)
    self.mgr_hi.free(weight.gate)
    self.mgr_hi.free(weight.up)
    self.mgr_ih.free(weight.down)
