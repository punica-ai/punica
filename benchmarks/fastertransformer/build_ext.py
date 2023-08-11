from benchmarks.fastertransformer import _build_ext

if __name__ == "__main__":
  module = _build_ext(do_cmake=True)
  print("Built ext:", module.__spec__.origin)
