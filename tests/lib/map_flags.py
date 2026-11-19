import sys

from sailify.nvcc_hgcc_options import translate_nvcc_options


def translate(args):
    return translate_nvcc_options(args)


def main(argv):
    return 0 if sys.stdout.write(" ".join(translate(argv)) + chr(10)) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
