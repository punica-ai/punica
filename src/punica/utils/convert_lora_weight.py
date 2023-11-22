import argparse
import re

import torch


def convert_lora_weight(peft_weight_path):
    weights = torch.load(
        peft_weight_path, map_location=torch.device("cpu"), weights_only=True
    )
    projs = set()
    num_layers = 0
    rank = 0
    tmp = {}
    for key, value in weights.items():
        layer, proj, ab = re.findall(
            r"\.(\d+)\..*\.(\w+)_proj\.lora_(A|B)\.weight$", key
        )[0]
        ab = ab.upper()
        layer = int(layer)
        projs.add(proj)
        # PyTorch Linear layer is column-major
        if ab == "A":
            assert value.size(0) < value.size(1)
            r = value.size(0)
        elif ab == "B":
            assert value.size(0) > value.size(1)
            r = value.size(1)
        else:
            raise KeyError(f"Unknown weight key: {key}")
        if rank != 0:
            assert r == rank
        else:
            rank = r
        num_layers = max(num_layers, layer + 1)
        tmp[(layer, proj, ab)] = value

    out = {}
    for proj in projs:
        for ab in "AB":
            tensors = []
            for layer in range(num_layers):
                tensors.append(tmp[(layer, proj, ab)])
            out[f"{proj}.{ab}"] = torch.stack(tensors)

    return out


def _convert_lora_weight_main():
    parser = argparse.ArgumentParser(
        description="Convert LoRA weight to Punica's format."
    )
    parser.add_argument(
        "input", help="Path to the LoRA weight file, as trained by PEFT."
    )
    parser.add_argument(
        "output", help="Path to the output LoRA weight in Punica's format."
    )
    args = parser.parse_args()

    weights = convert_lora_weight(args.input)
    print("Input:", args.input)
    for key, value in weights.items():
        print("Key:", key, " shape:", list(value.shape), " dtype:", value.dtype)
    torch.save(weights, args.output)
    print("Saved to:", args.output)


if __name__ == "__main__":
    _convert_lora_weight_main()
