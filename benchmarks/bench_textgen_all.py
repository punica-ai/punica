import gzip
import itertools
import json
import multiprocessing
import pathlib
from datetime import datetime

import pytz

from .bench_textgen import *


def _run_in_subprocess_worker(q, target, args, kwargs):
  q.put(target(*args, **kwargs))


def run_in_subprocess(target, args=(), kwargs={}):
  q = multiprocessing.Queue()
  p = multiprocessing.Process(
      target=_run_in_subprocess_worker, args=(q, target, args, kwargs))
  p.start()
  p.join()
  return q.get()


def _get_device_name(device):
  return torch.cuda.get_device_name(torch.device(device))


def bench_all():
  project_root = pathlib.Path(__file__).parents[1]
  local_now = datetime.now(pytz.timezone("US/Pacific"))
  outpath = project_root / "data" / f"{local_now:%Y%m%d-%H%M%S}-bench_textgen.json.gz"
  outpath.parent.mkdir(parents=True, exist_ok=True)

  rs = generate_request_set(num_requests=100, maxlen=2048)
  batch_sizes = list(reversed(list(range(1, 17))))
  model = "7b"
  dtype = "float16"
  device = "cuda:0"
  device_name = run_in_subprocess(_get_device_name, args=(device,))
  expcfgs = list(itertools.product(batch_sizes, BENCH_FN.keys()))

  experiments = []
  out = {
      "requests": {
          "num_requests": len(rs),
          "prompt_lens": rs.prompt_lens.astype(int).tolist(),
          "output_lens": rs.output_lens.astype(int).tolist(),
          "total_prompt_tokens": int(rs.prompt_lens.sum()),
          "total_output_tokens": int(rs.output_lens.sum()),
      },
      "setup": {
          "device_name": device_name,
          "timestamp": int(local_now.timestamp()),
      },
      "experiments": experiments,
  }
  with gzip.open(outpath, "wt") as f:
    json.dump(out, f)

  for cfg_id, (bs, system) in enumerate(expcfgs, start=1):
    bench_fn = BENCH_FN[system]
    model_cfg = MODEL_CFGS[model]
    model_cfg.dtype = dtype
    model_cfg.device = device
    textgen_cfg = TextGenConfig(batch_size=bs)
    config = {
        "batch_size": bs,
        "system": system,
        "model": model,
        "dtype": dtype,
    }

    print(f"[{cfg_id}/{len(expcfgs)}] Running", config)
    res = run_in_subprocess(bench_fn, (model_cfg, textgen_cfg, rs))

    result = {
        "encode_latency":
            res.encode_latency.tolist(),
        "decode_latency":
            res.decode_latency.tolist(),
        "duration":
            res.duration,
        "avg_encode_latency": (res.encode_latency / rs.prompt_lens).mean(),
        "avg_decode_latency": (res.decode_latency / rs.output_lens).mean(),
        "throughput":
            (rs.prompt_lens.sum() + rs.output_lens.sum()) / res.duration,
    }
    experiments.append({"config": config, "result": result})
    with gzip.open(outpath, "wt") as f:
      json.dump(out, f)


if __name__ == "__main__":
  bench_all()
