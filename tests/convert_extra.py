import os
import shutil
import sys

from sailify.sailify_python import sailify

def main(argv):
    if len(argv) < 1:
        sys.stderr.write("usage: convert_extra.py OUTPUT_DIR [CASES_DIR]")
        return 2
    out_dir = argv[0]
    here = os.path.dirname(os.path.abspath(__file__))
    cases_dir = argv[1] if len(argv) > 1 else os.path.join(here, "cases")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    results = sailify(
        project_directory=cases_dir,
        output_directory=out_dir,
        verbose=True,
    )
    if isinstance(results, dict):
        items = sorted(results.items())
    else:
        items = [(str(i), r) for i, r in enumerate(results)]
    ok = 0
    failed = 0
    for k, r in items:
        status = getattr(r, "status", "ok")
        ppu = getattr(r, "ppuified_path", "")
        print("RESULT %s status=%s out=%s" % (os.path.basename(str(k)), status, ppu))
        if status == "ok":
            ok += 1
        else:
            failed += 1
    print("converted=%d failed=%d" % (ok, failed))
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
