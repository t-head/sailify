import sys

from sailify.nvcc_hgcc_options import NVCC_HGCC_OPTIONS as OPT


def translate(args):
    out = []
    i = 0
    n = len(args)
    while i < n:
        a = args[i]
        e = OPT.get(a)
        if e is None:
            out.append(a)
            i += 1
            continue
        if e.get("needs_ignore"):
            i += 1
            if e.get("needs_value"):
                i += 1
            continue
        m = e.get("map_to")
        if m is None:
            i += 1
            continue
        if isinstance(m, list):
            out.extend(m)
        else:
            out.append(m)
            if e.get("needs_value") and i + 1 < n:
                i += 1
                v = args[i]
                vm = e.get("value_map") or {}
                out.append(vm.get(v, v))
        i += 1
    return out


def main(argv):
    return 0 if sys.stdout.write(" ".join(translate(argv)) + chr(10)) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
