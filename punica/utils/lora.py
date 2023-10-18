from typing import Sequence

import torch

__all__ = [
    "LoraWeight",
    "BatchedLoraWeight",
]


class LoraWeight:

  def __init__(
      self,
      num_layers: int,
      in_features: int,
      out_features: int,
      lora_rank: int,
      dtype: torch.dtype,
      device: torch.device,
  ):
    self.wa = torch.empty((num_layers, in_features, lora_rank),
                          dtype=dtype,
                          device=device)
    self.wb = torch.empty((num_layers, lora_rank, out_features),
                          dtype=dtype,
                          device=device)

  @property
  def device(self) -> torch.device:
    return self.wa.device

  @property
  def dtype(self) -> torch.dtype:
    return self.wa.dtype

  @property
  def num_layers(self) -> int:
    return self.wa.size(0)

  @property
  def in_features(self) -> int:
    return self.wa.size(1)

  @property
  def out_features(self) -> int:
    return self.wb.size(2)

  @property
  def lora_rank(self) -> int:
    return self.wa.size(2)


class BatchedLoraWeight:

  def __init__(self, weights: Sequence[LoraWeight]):
    assert len(weights) > 0
    device = weights[0].device
    self.wa_ptr = torch.tensor([w.wa.data_ptr() for w in weights],
                               dtype=torch.int64,
                               device=device)
    self.wb_ptr = torch.tensor([w.wb.data_ptr() for w in weights],
                               dtype=torch.int64,
                               device=device)
