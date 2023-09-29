import pytest
import torch

import punica.ops


def assert_close(a, b):
  rtol, atol = {
      torch.float16: (5e-3, 5e-3),
      torch.bfloat16: (3e-2, 2e-2),
  }[a.dtype]
  torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


def _rms_norm_ref_impl(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float = 1e-6,
):
  dtype = x.dtype
  x = x.to(torch.float32)
  variance = x.pow(2).mean(-1, keepdim=True)
  x = x * torch.rsqrt(variance + eps)
  return (w * x).to(dtype)


@pytest.mark.parametrize("dtype_str", ["float16", "bfloat16"])
@torch.inference_mode()
def test_rms_norm_correctness(dtype_str):
  torch.manual_seed(0xabcdabcd987)
  h = 4096
  bs = 17
  dtype = getattr(torch, dtype_str)
  device = torch.device("cuda:0")

  w = torch.randn(h, dtype=dtype, device=device)
  x = torch.randn(bs, h, dtype=dtype, device=device)

  y_ref = _rms_norm_ref_impl(x, w)
  y_our = punica.ops.rms_norm(x, w)
  torch.testing.assert_close(y_ref, y_our)
