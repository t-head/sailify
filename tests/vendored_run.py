import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))
from map_flags import translate

_RELEASE_VER_RE = re.compile(r"Release version (\d+)\.(\d+)")
_MIN_SDK_INT = 20200


def _sdk_version_int(hgcc):
    try:
        proc = subprocess.run(
            [hgcc, "--version"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = _RELEASE_VER_RE.search(proc.stdout or "")
    if not m:
        return None
    return int(m.group(1)) * 10000 + int(m.group(2)) * 100


def _load_min_sdk_cases(vendored_dir):
    path = os.path.join(vendored_dir, "needs_ppu_sdk_2v2.txt")
    cases = set()
    if os.path.isfile(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    cases.add(line)
    return cases


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendored-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hgcc", required=True)
    ap.add_argument("--cases-log", required=True)
    args = ap.parse_args(argv)

    from sailify.sailify_python import sailify

    sdk_int = _sdk_version_int(args.hgcc)
    min_sdk_cases = _load_min_sdk_cases(args.vendored_dir)
    skip_set = min_sdk_cases if (sdk_int is not None and sdk_int < _MIN_SDK_INT) else set()
    if skip_set:
        print("vendored: PPU SDK %d < 2.2, skipping %d cases (needs_ppu_sdk_2v2.txt)"
              % (sdk_int, len(skip_set)))

    os.makedirs(args.out, exist_ok=True)
    cl = open(args.cases_log, "w")
    inc_common = os.path.join(args.vendored_dir, "include")
    npass = 0
    nfail = 0
    nskip = 0
    for cat in sorted(os.listdir(args.vendored_dir)):
        catdir = os.path.join(args.vendored_dir, cat)
        if not os.path.isdir(catdir):
            continue
        srcs = []
        for root, dirs, files in os.walk(catdir):
            for f in files:
                if f.endswith(".cu"):
                    srcs.append(os.path.join(root, f))
        if not srcs:
            continue
        flags = []
        fpath = os.path.join(catdir, "flags.txt")
        if os.path.isfile(fpath):
            with open(fpath) as fh:
                flags = fh.read().split()
        outdir = os.path.join(args.out, cat)
        results = sailify(project_directory=catdir, output_directory=outdir)
        conv = []
        for root, dirs, files in os.walk(outdir):
            for f in files:
                if f.endswith(".cu"):
                    conv.append(os.path.join(root, f))
        got = set(os.path.basename(f) for f in conv)
        for s in sorted(srcs):
            b = os.path.basename(s)
            if b not in got:
                cl.write("FAIL convert vendored " + cat + "/" + b[:-3] + " (no converted output)" + chr(10))
                nfail += 1
        incs = ["-I" + outdir, "-I" + catdir, "-I" + inc_common]
        compat = os.path.join(outdir, ".ppu_compat", "compatible_wrapper.h")
        wrap = []
        if os.path.isfile(compat):
            incs.append("-I" + os.path.join(outdir, ".ppu_compat"))
            wrap = ["-include", "compatible_wrapper.h"]
        wrap = wrap + ["-include", "gtest/gtest.h"]
        hflags = translate(["-c", "-O2", "-std=c++14", "-D__USE_PPUSDK__"] + flags)
        for f in sorted(conv):
            rel = os.path.relpath(f, outdir)[:-3]
            key = cat + "/" + rel
            if key in skip_set:
                cl.write("SKIP compile vendored " + key + " (needs PPU SDK >= 2.2)" + chr(10))
                nskip += 1
                continue
            obj = os.path.join(args.out, "obj_" + cat + "_" + rel.replace("/", "_") + ".o")
            cmd = [args.hgcc] + hflags + incs + wrap + [f, "-o", obj]
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if p.returncode == 0:
                cl.write("PASS compile vendored " + cat + "/" + rel + chr(10))
                npass += 1
            else:
                logf = os.path.join(args.out, "log_" + cat + "_" + rel.replace("/", "_") + ".txt")
                with open(logf, "wb") as lf:
                    lf.write(p.stdout or b"")
                cl.write("FAIL compile vendored " + cat + "/" + rel + " (see " + logf + ")" + chr(10))
                nfail += 1
    cl.write("VENDORED_SUMMARY pass=" + str(npass) + " fail=" + str(nfail)
             + " skip=" + str(nskip) + chr(10))
    cl.close()
    print("VENDORED_SUMMARY pass=" + str(npass) + " fail=" + str(nfail)
          + " skip=" + str(nskip))
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
