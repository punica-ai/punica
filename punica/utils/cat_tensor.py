from typing import Sequence

import torch


class CatTensor:

  def __init__(self, cat: torch.Tensor, lens: Sequence[int]):
    assert cat.size(0) == sum(lens)
    self._cat = cat
    self._lens = tuple(lens)

  @property
  def cat(self):
    return self._cat

  @property
  def lens(self):
    return self._lens

  def split(self):
    return self._cat.split(self._lens)
