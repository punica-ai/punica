import contextlib
import datetime
import itertools
import os
import pathlib
import platform
import re
import subprocess

import setuptools
import torch
import torch.utils.cpp_extension as torch_cpp_ext

root = pathlib.Path(__name__).parent


def glob(pattern):
    return [str(p) for p in root.glob(pattern)]


def remove_unwanted_pytorch_nvcc_flags():
    REMOVE_NVCC_FLAGS = [
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    for flag in REMOVE_NVCC_FLAGS:
        with contextlib.suppress(ValueError):
            torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)


def generate_flashinfer_cu() -> list[str]:
    page_sizes = os.environ.get("PUNICA_PAGE_SIZES", "16").split(",")
    group_sizes = os.environ.get("PUNICA_GROUP_SIZES", "1,2,4,8").split(",")
    head_dims = os.environ.get("PUNICA_HEAD_DIMS", "128").split(",")
    page_sizes = [int(x) for x in page_sizes]
    group_sizes = [int(x) for x in group_sizes]
    head_dims = [int(x) for x in head_dims]
    dtypes = {"fp16": "nv_half", "bf16": "nv_bfloat16"}
    funcs = ["prefill", "decode"]
    prefix = "csrc/flashinfer_adapter/generated"
    (root / prefix).mkdir(parents=True, exist_ok=True)
    files = []

    # dispatch.inc
    path = root / prefix / "dispatch.inc"
    if not path.exists():
        with open(root / prefix / "dispatch.inc", "w") as f:
            f.write("#define _DISPATCH_CASES_page_size(...)       \\\n")
            for x in page_sizes:
                f.write(f"  _DISPATCH_CASE({x}, PAGE_SIZE, __VA_ARGS__) \\\n")
            f.write("// EOL\n")

            f.write("#define _DISPATCH_CASES_group_size(...)      \\\n")
            for x in group_sizes:
                f.write(f"  _DISPATCH_CASE({x}, GROUP_SIZE, __VA_ARGS__) \\\n")
            f.write("// EOL\n")

            f.write("#define _DISPATCH_CASES_head_dim(...)        \\\n")
            for x in head_dims:
                f.write(f"  _DISPATCH_CASE({x}, HEAD_DIM, __VA_ARGS__) \\\n")
            f.write("// EOL\n")

            f.write("\n")

    # impl
    for func, page_size, group_size, head_dim, dtype in itertools.product(
        funcs, page_sizes, group_sizes, head_dims, dtypes
    ):
        fname = f"batch_{func}_p{page_size}_g{group_size}_h{head_dim}_{dtype}.cu"
        files.append(prefix + "/" + fname)
        if (root / prefix / fname).exists():
            continue
        with open(root / prefix / fname, "w") as f:
            f.write('#include "../flashinfer_decl.h"\n\n')
            f.write(f'#include "flashinfer/{func}.cuh"\n\n')
            f.write(
                f"INST_Batch{func.capitalize()}({dtypes[dtype]}, {page_size}, {group_size}, {head_dim})\n"
            )

    return files


def get_local_version_suffix() -> str:
    if not (root / ".git").is_dir():
        return ""
    now = datetime.datetime.now()
    git_hash = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True
    ).strip()
    commit_number = subprocess.check_output(
        ["git", "rev-list", "HEAD", "--count"], cwd=root, text=True
    ).strip()
    dirty = ".dirty" if subprocess.run(["git", "diff", "--quiet"]).returncode else ""
    return f"+c{commit_number}.d{now:%Y%m%d}.{git_hash}{dirty}"


def get_version() -> str:
    version = os.getenv("PUNICA_BUILD_VERSION")
    if version is None:
        with open(root / "version.txt") as f:
            version = f.read().strip()
        version += get_local_version_suffix()
    return version


def get_cuda_version() -> tuple[int, int]:
    if torch_cpp_ext.CUDA_HOME is None:
        nvcc = "nvcc"
    else:
        nvcc = os.path.join(torch_cpp_ext.CUDA_HOME, "bin/nvcc")
    txt = subprocess.check_output([nvcc, "--version"], text=True)
    major, minor = map(int, re.findall(r"release (\d+)\.(\d+),", txt)[0])
    return major, minor


def generate_build_meta() -> None:
    d = {}
    version = get_version()
    d["cuda_major"], d["cuda_minor"] = get_cuda_version()
    d["torch"] = torch.__version__
    d["python"] = platform.python_version()
    d["TORCH_CUDA_ARCH_LIST"] = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
    with open(root / "src/punica/_build_meta.py", "w") as f:
        f.write(f"__version__ = {version!r}\n")
        f.write(f"build_meta = {d!r}")


if __name__ == "__main__":
    remove_unwanted_pytorch_nvcc_flags()
    generate_build_meta()

    ext_modules = []
    ext_modules.append(
        torch_cpp_ext.CUDAExtension(
            name="punica.ops._kernels",
            sources=[
                "csrc/punica_ops.cc",
                "csrc/bgmv/bgmv_all.cu",
                "csrc/flashinfer_adapter/flashinfer_all.cu",
                "csrc/rms_norm/rms_norm_cutlass.cu",
                "csrc/sgmv/sgmv_cutlass.cu",
                "csrc/sgmv_flashinfer/sgmv_all.cu",
            ]
            + generate_flashinfer_cu(),
            include_dirs=[
                str(root.resolve() / "third_party/cutlass/include"),
                str(root.resolve() / "third_party/flashinfer/include"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        )
    )

    setuptools.setup(
        version=get_version(),
        ext_modules=ext_modules,
        cmdclass={"build_ext": torch_cpp_ext.BuildExtension},
    )
