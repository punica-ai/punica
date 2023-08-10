from .fastertransformer.build_ext import build_ext as _build_ft


def main():
  ft = _build_ft()
  ft.run()


if __name__ == "__main__":
  main()
