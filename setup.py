import pathlib

import setuptools
import torch.utils.cpp_extension as torch_cpp_ext

root = pathlib.Path(__name__).parent


def glob(pattern):
  return [str(p) for p in root.glob(pattern)]


def get_version(path):
  with open(path) as f:
    for line in f:
      if line.startswith("__version__"):
        return line.split("=", maxsplit=1)[1].replace('"', '').strip()
  raise ValueError("Version not found")


def remove_unwanted_pytorch_nvcc_flags():
  REMOVE_NVCC_FLAGS = [
      '-D__CUDA_NO_HALF_OPERATORS__',
      '-D__CUDA_NO_HALF_CONVERSIONS__',
      '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
      '-D__CUDA_NO_HALF2_OPERATORS__',
  ]
  for flag in REMOVE_NVCC_FLAGS:
    try:
      torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)
    except ValueError:
      pass


remove_unwanted_pytorch_nvcc_flags()
ext_modules = []
ext_modules.append(
    torch_cpp_ext.CUDAExtension(
        "punica.ops._kernels",
        ["punica/ops/csrc/punica_ops.cc"] + glob("punica/ops/csrc/gen/*.cu") +
        glob("punica/ops/csrc/bgmv/*.cu") +
        glob("punica/ops/csrc/flashinfer_adapter/*.cu") +
        glob("punica/ops/csrc/rms_norm/*.cu"),
    ))

setuptools.setup(
    name="punica",
    version=get_version(root / "punica/__init__.py"),
    author="Punica Team",
    license="Apache License 2.0",
    description="Punica: System for serving Large Language Models",
    url="https://github.com/abcdabcd987/punica",
    packages=setuptools.find_packages(
        exclude=("assets", "benchmarks", "csrc", "docs", "examples", "tests")),
    python_requires=">=3.8",
    ext_modules=ext_modules,
    cmdclass={"build_ext": torch_cpp_ext.BuildExtension},
)
