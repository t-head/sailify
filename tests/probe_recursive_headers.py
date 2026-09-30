import os, sys, tempfile, shutil
proj=sys.argv[1] if len(sys.argv)>1 else os.path.join(os.path.dirname(os.path.abspath(__file__)),"probe_proj")
out=tempfile.mkdtemp(prefix="rec_")
try:
    from sailify.sailify_python import sailify
    sailify(project_directory=proj, output_directory=out)
except Exception as e:
    print("RECURSIVE_PROBE sailify_failed", type(e).__name__)
    shutil.rmtree(out,ignore_errors=True)
    sys.exit(3)
fh=[]
fc=[]
for dp,dn,fn in os.walk(out):
    for f in fn:
        if f.endswith(".h"): fh.append(os.path.relpath(os.path.join(dp,f),out))
        if f.endswith(".cu"): fc.append(os.path.relpath(os.path.join(dp,f),out))
names=[os.path.basename(x) for x in fh]
want=["p_hdr01.h","p_hdr02.h","p_hdr03.h"]
present=len([w for w in want if w in names])
print("RECURSIVE_PROBE headers_in_output", present, "of", len(want))
print("RECURSIVE_PROBE headers_list", fh)
print("RECURSIVE_PROBE cu_in_output", len(fc))
shutil.rmtree(out,ignore_errors=True)
sys.exit(0)
