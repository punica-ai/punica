from collections.abc import Sequence

import numpy as np
import torch


class BatchLenInfo:
    def __init__(
        self,
        prefills: Sequence[int],
        decode: int,
        indptr_device: torch.device,
        indptr_dtype: torch.dtype = torch.int32,
    ):
        tmp = [0]
        tmp.extend(prefills)
        self._prefills = tmp[1:]
        self._decode = decode
        if len(prefills) > 0:
            cumsum = np.cumsum(tmp)
            self._indptr = torch.tensor(
                cumsum, dtype=indptr_dtype, device=indptr_device
            )
            self._doff = cumsum[-1]
        else:
            self._indptr = None
            self._doff = 0

    @property
    def prefills(self) -> list[int]:
        """Length of each prefill request."""
        return self._prefills

    @property
    def decode(self) -> int:
        """Number of decode requests."""
        return self._decode

    @property
    def doff(self) -> int:
        """Index of the first decode request. Equivalently, total length of prefills."""
        return self._doff

    @property
    def indptr(self) -> torch.Tensor | None:
        """`indptr[i] := sum(prefills[:i])`. None if no prefill."""
        return self._indptr
