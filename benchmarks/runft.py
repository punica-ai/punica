from .fastertransformer import build_ext as _build_ft

ctr = 0


def cb():
  global ctr
  ctr += 1
  print("hi", ctr)


def main():
  ft = _build_ft()
  model = ft.FtLlama(
      num_heads=32,
      head_dim=128,
      inter_size=11008,
      num_layers=32,
      dtype="float16",
  )
  input_ids = [
      [0, 37, 92, 26, 66, 36, 55, 70, 73, 15, 36, 51, 34, 52, 29],
      [0, 37, 92, 70, 73, 15, 66, 93, 34, 52, 99],
      [0, 92, 16, 66, 16, 45, 70, 93, 11, 36, 53, 30, 52, 29],
      [0, 37, 92, 26, 66, 36, 55, 70, 23, 23],
      [0, 37, 92, 26, 66, 36, 55, 70, 29, 15, 34, 52, 23],
      [0, 73],
      [0, 37, 92, 15, 66, 93, 34, 52, 23],
      [0, 70, 73, 15, 66, 93, 30, 92, 29],
  ]
  model.forward(input_ids, 20, cb)


if __name__ == "__main__":
  main()
