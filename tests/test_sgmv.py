import numpy as np
import pytest
import torch

import punica.ops


def assert_close(a, b):
    rtol, atol = {
        torch.float16: (5e-3, 5e-3),
        torch.bfloat16: (3e-2, 2e-2),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


def sgmv_ref_impl(
    y: torch.Tensor,
    x: torch.Tensor,
    w: list[torch.Tensor],
    s: torch.Tensor,
    layer_idx: int,
):
    for i in range(len(w)):
        xi = x[s[i] : s[i + 1]].to(torch.float32)
        wi = w[i][layer_idx, :, :].T.to(torch.float32)
        yi = y[s[i] : s[i + 1]].to(torch.float32)
        y[s[i] : s[i + 1]] = (yi + xi @ wi).to(y.dtype)


def get_lora_lens(bs: int, popularity: str) -> list[int]:
    if popularity == "identical":
        return [bs]
    if popularity == "distinct":
        return [1] * bs
    if popularity == "uniform":
        n = int(np.ceil(np.sqrt(bs)))
        lens = np.array([bs // n] * n)
        while True:
            diff = bs - lens.sum()
            if diff == 0:
                break
            lens[: abs(diff)] += np.sign(diff)
        return lens.tolist()
    if popularity.startswith("zipf:"):
        alpha = float(popularity.split(":")[1])
        assert alpha > 1
        lens = []
        a = 1
        while sum(lens) + int(np.floor(a)) < bs:
            lens.append(int(np.floor(a)))
            a *= alpha
        lens.append(bs - sum(lens))
        return sorted(lens, reverse=True)
    if popularity.startswith("skewed"):
        if bs < 3:
            return [bs]
        # Create a highly imbalanced distribution by setting the first segment
        # length to 1 and the remainder to the second segment.
        return [1, bs - 1]
    raise KeyError(popularity)


def lora_ref_impl(
    y: torch.Tensor,
    x: torch.Tensor,
    wa: torch.Tensor,
    wb: torch.Tensor,
    s: torch.IntTensor,
    layer_idx: int,
):
    for i in range(len(wa)):
        xi = x[s[i] : s[i + 1]].to(torch.float32)
        wai = wa[i][layer_idx, :, :].to(torch.float32)
        wbi = wb[i][layer_idx, :, :].to(torch.float32)
        yi = y[s[i] : s[i + 1]].to(torch.float32)
        tmp = (xi @ wai).to(y.dtype).to(torch.float32)
        y[s[i] : s[i + 1]] = (yi + tmp @ wbi).to(y.dtype)


@pytest.mark.parametrize("dtype_str", ["float16", "bfloat16"])
@pytest.mark.parametrize("h", [4096, 11008])
@pytest.mark.parametrize("r", [16, 32, 64, 96, 128])
@pytest.mark.parametrize(
    "direction",
    [
        "shrink",
        pytest.param("expand", marks=pytest.mark.xfail(reason="TODO: sgmv expand")),
    ],
)
@pytest.mark.parametrize("popularity", ["distinct", "uniform", "zipf:1.5", "identical", "skewed"])
@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 7, 10, 16, 32, 64, 133])
@torch.inference_mode()
def test_sgmv_correctness(dtype_str, h, r, direction, popularity, batch_size):
    torch.manual_seed(0xABCDABCD987)
    seqlens = get_lora_lens(batch_size, popularity)
    num_layers = 5
    dtype = getattr(torch, dtype_str)
    device = torch.device("cuda:0")
    if direction == "shrink":
        h1, h2 = h, r
    else:
        h1, h2 = r, h

    w = [
        torch.randn((num_layers, h2, h1), dtype=dtype, device=device)
        for _ in range(len(seqlens))
    ]
    w_ptr = torch.tensor([t.data_ptr() for t in w], dtype=torch.int64, device=device)
    s = torch.cumsum(
        torch.tensor([0] + seqlens, device=device),
        dim=0,
        dtype=torch.int32,
    )
    x = torch.randn((int(s[-1]), h1), dtype=dtype, device=device)
    y = torch.randn((int(s[-1]), h2), dtype=dtype, device=device)
    for layer_idx in range(num_layers):
        y_ref = y.clone()
        sgmv_ref_impl(y_ref, x, w, s, layer_idx)
        y_our = y.clone()
        punica.ops.sgmv(y_our, x, w_ptr, s, layer_idx)
        assert_close(y_ref, y_our)


@pytest.mark.xfail(reason="TODO: sgmv expand")
@pytest.mark.parametrize("dtype_str", ["float16", "bfloat16"])
@pytest.mark.parametrize("batch_setup", ["1x7", "7x1", "3x3", "32x1", "1x32"])
@torch.inference_mode()
def test_lora_correctness(dtype_str, batch_setup):
    torch.manual_seed(0xABCDABCD987)
    num_layers = 5
    h1 = 4096
    h2 = 11008
    r = 16
    num_problems, problem_size = map(int, batch_setup.split("x"))
    dtype = getattr(torch, dtype_str)
    device = torch.device("cuda:0")

    wa = [
        torch.rand((num_layers, h1, r), dtype=dtype, device=device)
        for _ in range(num_problems)
    ]
    wb = [
        torch.rand((num_layers, r, h2), dtype=dtype, device=device)
        for _ in range(num_problems)
    ]
    wa_ptr = torch.tensor([t.data_ptr() for t in wa], dtype=torch.int64, device=device)
    wb_ptr = torch.tensor([t.data_ptr() for t in wb], dtype=torch.int64, device=device)
    s = torch.cumsum(
        torch.tensor([0] + [problem_size] * num_problems, device=device),
        dim=0,
        dtype=torch.int32,
    )
    x = torch.rand((s[-1], h1), dtype=dtype, device=device)
    y = torch.rand((s[-1], h2), dtype=dtype, device=device)

    for layer_idx in range(num_layers):
        y_ref = y.clone()
        lora_ref_impl(y_ref, x, wa, wb, s, layer_idx)
        y_our = y.clone()
        punica.ops.add_lora_bgmv(y_our, x, wa_ptr, wb_ptr, s, layer_idx, r)
        assert_close(y_ref, y_our)


@pytest.mark.parametrize(
    "direction",
    [
        "shrink",
        pytest.param("expand", marks=pytest.mark.xfail(reason="TODO: sgmv expand")),
    ],
)
@torch.inference_mode()
def test_sgmv_cuda_graph(direction):
    torch.manual_seed(0xABCDABCD987)
    batch_size = 133
    popularity = "zipf:1.5"
    seqlens = get_lora_lens(batch_size, popularity)
    num_layers = 5
    dtype = torch.float16
    device = torch.device("cuda:0")
    h, r = 11008, 16
    if direction == "shrink":
        h1, h2 = h, r
    else:
        h1, h2 = r, h

    w = [
        torch.randn((num_layers, h2, h1), dtype=dtype, device=device)
        for _ in range(len(seqlens))
    ]
    w_ptr = torch.tensor([t.data_ptr() for t in w], dtype=torch.int64, device=device)
    s = torch.cumsum(
        torch.tensor([0] + seqlens, device=device),
        dim=0,
        dtype=torch.int32,
    )
    x = torch.randn((int(s[-1]), h1), dtype=dtype, device=device)
    y = torch.randn((int(s[-1]), h2), dtype=dtype, device=device)
    for layer_idx in range(num_layers):
        y_our = y.clone()
        punica.ops.sgmv(y_our, x, w_ptr, s, layer_idx)

        y_graph = torch.empty_like(y)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            punica.ops.sgmv(y_graph, x, w_ptr, s, layer_idx)

        for _ in range(2):
            y_graph.copy_(y.clone())
            graph.replay()
            assert (y_graph == y_our).all()
