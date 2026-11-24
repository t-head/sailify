"""Tests for custom_impl.json stub customization (sailify.unsupported_stubs)."""

import logging
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, os.path.abspath(REPO))

from sailify.unsupported_stubs import generate_stubs as G


def _report(npass, nfail):
    print("CUSTOM_IMPL_SUMMARY pass=" + str(npass) + " fail=" + str(nfail))
    return 0 if nfail == 0 else 1


_API = {
    "name": "acblasSetKernelStream",
    "return_type": "acblasStatus_t",
    "parameters": [{"name": "stream", "type": "hggcStream_t {}"}],
    "library": "cublas",
}


def test_default_stub_exits():
    """Unsupported stubs print an error and exit(1) regardless of return type."""
    failures = []
    stub = G._generate_stub_function(_API)
    if "exit(1)" not in stub:
        failures.append("default stub no longer exits: %r" % stub)
    if "ACBLAS_STATUS_NOT_SUPPORTED" in stub:
        failures.append("per-return-type status leaked into default stub")
    if "return (acblasStatus_t)0;" not in stub:
        failures.append("unreachable return missing: %r" % stub)
    return failures


def test_custom_stub_body_verbatim():
    """The impl is inserted verbatim: blank lines and spacing are preserved."""
    failures = []
    impl = ('    fprintf(stderr, "acblas  fallback");\n\n'
            '    return   ACBLAS_STATUS_SUCCESS;')
    stub = G._generate_stub_function(_API, impl)
    if impl not in stub:
        failures.append("custom impl not inserted verbatim: %r" % stub)
    if "exit(1)" in stub or "is not supported." in stub:
        failures.append("custom stub kept default fatal body: %r" % stub)
    if "(void)stream;" not in stub:
        failures.append("custom stub missing unused-parameter casts: %r" % stub)
    return failures


def test_packaged_config_applied_in_header():
    """The shipped custom_impl.json customizes the cusparse gtsv2 stubs."""
    failures = []
    hdr = G.generate_stubs_header(stub_probe=False)
    if "return ACSPARSE_STATUS_NOT_SUPPORTED;" not in hdr:
        failures.append("cusparse custom impl missing from generated header")
    if "acsparseSgtsv2StridedBatch is not supported." in hdr:
        failures.append("customized acsparseSgtsv2StridedBatch kept the fatal body")
    plain = hdr.find("acblasAsumEx_64(")
    custom = hdr.find("acsparseSgtsv2StridedBatch(")
    if 0 <= custom < plain and "exit(1)" in hdr[custom:plain]:
        failures.append("customized stub body contains exit(1)")
    return failures


def test_unmatched_entries_silent():
    """Entries matching no generated stub are ignored without warnings."""
    failures = []
    saved = G._load_custom_impls
    records = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Collect(level=logging.WARNING)
    G.log.addHandler(handler)
    try:
        G._load_custom_impls = lambda: {
            ("cublas", "acblasNopeNotThere"): "return ACBLAS_STATUS_SUCCESS;",
        }
        G.generate_stubs_header(stub_probe=False)
    finally:
        G._load_custom_impls = saved
        G.log.removeHandler(handler)
    if records:
        failures.append("unmatched entry produced warnings: %r" % records)
    return failures


def main(argv=None):
    npass = 0
    nfail = 0
    for name, fn in [
        ("default_stub_exits", test_default_stub_exits),
        ("custom_stub_body_verbatim", test_custom_stub_body_verbatim),
        ("packaged_config_applied_in_header", test_packaged_config_applied_in_header),
        ("unmatched_entries_silent", test_unmatched_entries_silent),
    ]:
        try:
            failures = fn()
        except Exception as exc:
            failures = ["exception: %r" % (exc,)]
        if failures:
            nfail += 1
            for f in failures:
                print("FAIL custom_impl " + name + ": " + f)
        else:
            npass += 1
            print("PASS custom_impl " + name)
    return _report(npass, nfail)


if __name__ == "__main__":
    sys.exit(main())
