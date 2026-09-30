#!/usr/bin/env bash

set -u

SCRIPT_PATH=${0}
SCRIPT_DIR=${SCRIPT_PATH%/*}
REPO_DIR=${SCRIPT_DIR}/..
if [ ! -f "${REPO_DIR}/sailify_cli.py" ]; then
  echo "please run from the repo root: bash tests/run_tests.sh"
  exit 2
fi

PASS=0
FAIL=0
BPASS=0
BFAIL=0
SKIP_COMPILE=0

while [ -n "${1:-}" ]; do
  case "${1}" in
    --skip-compile)
      SKIP_COMPILE=1
      shift
      ;;
    *)
      echo "unknown option: ${1}"
      exit 2
      ;;
  esac
done

PPU_SDK=${PPU_SDK:-${SCRIPT_DIR}/../../../PPU_SDK}
if [ ! -f "${PPU_SDK}/envsetup.sh" ]; then
  echo "FAIL env: PPU SDK not found at ${PPU_SDK} (set PPU_SDK)"
  exit 2
fi

set +u
. "${PPU_SDK}/envsetup.sh"
set -u
if [ -z "${PPU_PATH:-}" ]; then
  echo "FAIL env: envsetup.sh did not set PPU_PATH"
  exit 2
fi
HGCC=${PPU_PATH}/bin/hgcc
if [ ! -x "${HGCC}" ]; then
  echo "FAIL env: hgcc not executable at ${HGCC}"
  exit 2
fi

HGVER_LINE=$("${HGCC}" --version 2>/dev/null | grep -m1 "Release version")
echo "env: ${HGVER_LINE:-PPU SDK version not parsed from hgcc --version}"

PYTHON=${PYTHON:-python3}
export PYTHONPATH=.
if "$PYTHON" -c "import sailify" 2>/dev/null; then
  echo "env: sailify import ok"
else
  echo "env: installing sailify in editable mode"
  "$PYTHON" -m pip install -e . >/dev/null 2>&1 || true
fi
if "$PYTHON" -c "import sailify" 2>/dev/null; then
  :
else
  echo "FAIL env: sailify not importable by ${PYTHON}"
  exit 2
fi

OUT=${TMPDIR:-/tmp}/sailify_tests_run
rm -rf "${OUT}"
mkdir -p "${OUT}/logs" "${OUT}/obj"

echo "== phase 1: CLI conversion of cli_project =="
if "$PYTHON" "${REPO_DIR}/sailify_cli.py" "${SCRIPT_DIR}/cli_project" --output-directory "${OUT}/cli" --verbose > "${OUT}/logs/cli_convert.log" 2>&1; then
  echo "PASS cli convert rc"
else
  echo "FAIL cli convert (see ${OUT}/logs/cli_convert.log)"
  let FAIL=FAIL+1
fi

CLI_MAIN=${OUT}/cli/src/main.cu
if [ -f "${CLI_MAIN}" ]; then
  echo "PASS cli output layout"
else
  echo "FAIL cli output missing main.cu"
  let FAIL=FAIL+1
fi
if [ -f "${OUT}/cli/.ppu_compat/compatible_wrapper.h" ]; then
  echo "PASS cli compat wrapper present"
else
  echo "WARN cli: .ppu_compat/compatible_wrapper.h not found"
fi

if grep -q "cudaMalloc" "${CLI_MAIN}"; then
  echo "WARN cli: raw cudaMalloc still present in converted main.cu"
else
  echo "PASS cli identifiers mapped"
fi

echo "== phase 2: Python API conversion =="
if "$PYTHON" "${SCRIPT_DIR}/convert_extra.py" "${OUT}/extra" > "${OUT}/logs/extra_convert.log" 2>&1; then
  echo "PASS extra convert"
else
  echo "FAIL extra convert (see ${OUT}/logs/extra_convert.log)"
  let FAIL=FAIL+1
fi

for c in hello_kernel vector_add shared_reduce ext_lambda events_api header_user; do
  find "${OUT}/extra" -name "${c}.cu" > "${OUT}/${c}.found"
  if [ -s "${OUT}/${c}.found" ]; then
    echo "PASS extra output ${c}.cu"
  else
    echo "FAIL extra output missing ${c}.cu"
    let FAIL=FAIL+1
  fi
done

if [ "${SKIP_COMPILE}" = "1" ]; then
  echo "SUMMARY pass=skip fail=$((FAIL+0))"
  if [ "${FAIL}" -gt 0 ]; then
    exit 1
  fi
  exit 0
fi

echo "== phase 3: compile converted extra cases with hgcc =="
"$PYTHON" "${SCRIPT_DIR}/lib/map_flags.py" -c -std=c++14 -O2 --extended-lambda > "${OUT}/flags.txt"
HGCC_FLAGS=""
while read -r a; do
  HGCC_FLAGS="${HGCC_FLAGS} ${a}"
done < "${OUT}/flags.txt"
echo "hgcc flags:${HGCC_FLAGS}"

INC="-I${OUT}/extra -I${OUT}/extra/headers -I${SCRIPT_DIR}/cases/headers"
WRAP=""
if [ -f "${OUT}/extra/.ppu_compat/compatible_wrapper.h" ]; then
  INC="${INC} -I${OUT}/extra/.ppu_compat"
  WRAP="-include compatible_wrapper.h"
else
  echo "WARN extra: no .ppu_compat, compiling without compatible_wrapper.h"
fi

for c in hello_kernel vector_add shared_reduce ext_lambda events_api header_user; do
  find "${OUT}/extra" -name "${c}.cu" > "${OUT}/${c}.found"
  cuf=""
  read -r cuf < "${OUT}/${c}.found" || true
  if [ -n "${cuf}" ] && "${HGCC}" ${HGCC_FLAGS} ${INC} ${WRAP} "${cuf}" -o "${OUT}/obj/${c}.o" 2> "${OUT}/logs/${c}.log"; then
    echo "PASS compile ${c}"
    let PASS=PASS+1
  else
    echo "FAIL compile ${c} (see ${OUT}/logs/${c}.log)"
    let FAIL=FAIL+1
  fi
done

echo "== phase 4: compile converted cli_project =="
INC2="-I${OUT}/cli"
WRAP2=""
if [ -f "${OUT}/cli/.ppu_compat/compatible_wrapper.h" ]; then
  INC2="${INC2} -I${OUT}/cli/.ppu_compat"
  WRAP2="-include compatible_wrapper.h"
fi
for pair in main:src/main.cu math_ops:src/util/math_ops.cu; do
  base=${pair%%:*}
  rel=${pair#*:}
  if "${HGCC}" ${HGCC_FLAGS} ${INC2} ${WRAP2} "${OUT}/cli/${rel}" -o "${OUT}/obj/cli_${base}.o" 2> "${OUT}/logs/cli_${base}.log"; then
    echo "PASS compile cli ${base}"
    let PASS=PASS+1
  else
    echo "FAIL compile cli ${base} (see ${OUT}/logs/cli_${base}.log)"
    let FAIL=FAIL+1
  fi
done

echo "== phase 5: vendored CompilerTest cases =="
VENDORED_DIR=${SCRIPT_DIR}/cases/compat
if [ -d "${VENDORED_DIR}" ]; then
  BRC=0
  "$PYTHON" "${SCRIPT_DIR}/vendored_run.py" --vendored-dir "${VENDORED_DIR}" --out "${OUT}/vendored" --hgcc "${HGCC}" --cases-log "${OUT}/logs/vendored_cases.log" > "${OUT}/logs/vendored.log" 2>&1 || BRC=1
  if [ -f "${OUT}/logs/vendored_cases.log" ]; then
    cat "${OUT}/logs/vendored_cases.log"
  fi
  BP=0
  BF=0
  grep "^VENDORED_SUMMARY" "${OUT}/logs/vendored.log" > "${OUT}/bs.txt" 2>/dev/null || true
  read -r _ BPW BFW _ < "${OUT}/bs.txt" || true
  BP=${BPW#pass=}
  BF=${BFW#fail=}
  let BPASS=BPASS+BP
  let BFAIL=BFAIL+BF
  let FAIL=FAIL+BRC
else
  echo "skip vendored (tests/cases/compat not found)"
fi
echo "== phase 6: real-world API patterns =="
APRC=0
"$PYTHON" "${SCRIPT_DIR}/api_patterns_test.py" > "${OUT}/logs/api_patterns.log" 2>&1 || APRC=1
cat "${OUT}/logs/api_patterns.log"
APP=0
APF=0
grep "^API_PATTERNS_SUMMARY" "${OUT}/logs/api_patterns.log" > "${OUT}/aps.txt" 2>/dev/null || true
read -r _ APPW APFW < "${OUT}/aps.txt" || true
APP=${APPW#pass=}
APF=${APFW#fail=}
let BPASS=BPASS+APP
let BFAIL=BFAIL+APF
let FAIL=FAIL+APRC
echo ""
echo "Artifacts and logs kept under: ${OUT}"

python3 ${SCRIPT_DIR}/probe_recursive_headers.py ${SCRIPT_DIR}/probe_proj || true

echo "SUMMARY pass=$((PASS+BPASS)) fail=$((FAIL+BFAIL))"
if [ "${FAIL}" -gt 0 ]; then
  exit 1
fi
exit 0
