import itertools
import pathlib
import subprocess
from datetime import datetime

import pytz


def main():
    this_file = pathlib.Path(__file__)
    project_root = this_file.parents[1]
    now = datetime.now(pytz.timezone("US/Pacific"))
    out_filename = f"{now:%Y%m%d-%H%M%S}-{this_file.stem}.jsonl"
    out_path = project_root / "data" / out_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    batch_size = 32
    num_batches = 32
    maxlen = 2048
    model_ = ["7b", "13b"]
    pop_ = ["bgmv", "bmm", "uniform", "zipf:1.5"]
    system_ = ["punica", "hf", "ds", "ft_backbone", "vllm_backbone"]
    all_ = list(itertools.product(model_, pop_, system_))

    for model, pop, system in all_:
        if system != "punica" and pop == "bgmv":
            continue
        args = {
            "--system": system,
            "--model": model,
            "--lora-popularity": pop,
            "--batch-size": str(batch_size),
            "--num-batches": str(num_batches),
            "--maxlen": str(maxlen),
            "--save-to": str(out_path),
        }
        cmd = ["python", "-m", "benchmarks.bench_textgen_lora"]
        cmd.extend([f"{k}={v}" for k, v in args.items()])
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=project_root)


if __name__ == "__main__":
    main()
