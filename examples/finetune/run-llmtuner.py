# ruff: noqa

import sys

sys.path.append("LLaMA-Factory/src")
from llmtuner import run_exp
from llmtuner.extras.template import register_template

register_template(
    name="empty",
    system="",
    prefix=["{{system}}"],
    prompt=["{{query}}"],
    sep=["\n"],
)

if __name__ == "__main__":
    run_exp()
