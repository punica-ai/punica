import argparse
import json
import pathlib

import datasets


def create_dataset(
    preset_name,
    dataset_name,
    dataset_config_name,
    prompt_template,
    response_template,
    output_dir,
):
    raw_datasets = datasets.load_dataset(dataset_name, dataset_config_name)
    for split, dataset in raw_datasets.items():
        filename = f"{preset_name}-{split}.jsonl"
        outpath = pathlib.Path(output_dir) / filename
        print(outpath)
        with open(outpath, "w") as f:
            for example in dataset:
                prompt = prompt_template.format(**example)
                response = response_template.format(**example)
                f.write(json.dumps({"prompt": prompt, "response": response}))
                f.write("\n")


presets = {}

presets["gsm8k"] = dict(
    dataset_name="gsm8k",
    dataset_config_name="main",
    prompt="<<SYS>>\nAnswer the following Grade School Math problem.\n<</SYS>>\n[INST] {question} [/INST]\n",
    response="{answer}",
)

presets["sqlctx"] = dict(
    dataset_name="b-mc2/sql-create-context",
    dataset_config_name="main",
    prompt="<<SYS>>\nGenerate a correct SQL query from the following database schema.\n{context}\n<</SYS>>\n[INST] {question} [/INST]\n",
    response="{answer}",
)

presets["viggo"] = dict(
    dataset_name="GEM/viggo",
    dataset_config_name="main",
    prompt="<<SYS>>\nGenerate a description based on the following representation.\n<</SYS>>\n[INST] {meaning_representation} [/INST]\n",
    response="{target}",
)


def main():
    data_dir = pathlib.Path(__file__).parent / "data"
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=presets.keys(), required=True)
    parser.add_argument("--output_dir", default=str(data_dir))
    args = parser.parse_args()
    p = presets[args.preset]
    create_dataset(
        preset_name=args.preset,
        dataset_name=p["dataset_name"],
        dataset_config_name=p["dataset_config_name"],
        prompt_template=p["prompt"],
        response_template=p["response"],
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
