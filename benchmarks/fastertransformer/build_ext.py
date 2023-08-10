import subprocess
import pathlib
import torch.utils.cpp_extension


def build_ext():
  project_root = pathlib.Path(__file__).parents[2].resolve()
  build_path = project_root / "build/fastertransformer"
  build_path.mkdir(parents=True, exist_ok=True)
  ext_name = "_punica_ft"

  if not (build_path / "CMakeCache.txt").exists():
    subprocess.check_call(
        [
            "cmake",
            "-DCMAKE_BUILD_TYPE=Release",
            f"{project_root}/benchmarks/fastertransformer",
        ],
        cwd=build_path,
    )

  if not (build_path / "libft.so").exists():
    nprocs = subprocess.check_output(["nproc"], text=True).strip()
    subprocess.check_call(
        ["cmake", "--build", ".", "--target", "ft", "--parallel", nprocs],
        cwd=build_path,
    )

  module = torch.utils.cpp_extension.load(
      name=ext_name,
      sources=[pathlib.Path(__file__).parent / "ft_pybind11.cc"],
      extra_ldflags=[
          f"{build_path}/libft.so",
          f"-Wl,-rpath={build_path}",
      ],
      build_directory=str(build_path),
  )
  return module


if __name__ == "__main__":
  module = build_ext()
  print("Built ext:", module.__spec__.origin)
