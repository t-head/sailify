"""Validate status returns, fallback behavior, and generated C/C++ stubs."""

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from sailify.unsupported_stubs import generate_stubs as generator


EXPECTED_STATUSES = {
    "acblasStatus_t": "ACBLAS_STATUS_NOT_SUPPORTED",
    "acdnnStatus_t": "ACDNN_STATUS_NOT_SUPPORTED",
    "acfftResult": "ACFFT_NOT_SUPPORTED",
    "acsolverStatus_t": "ACSOLVER_STATUS_NOT_SUPPORTED",
    "acsparseStatus_t": "ACSPARSE_STATUS_NOT_SUPPORTED",
    "HGresult": "HGGC_ERROR_NOT_SUPPORTED",
    "hggcError_t": "hggcErrorNotSupported",
    "hgmlReturn_t": "HGML_ERROR_NOT_SUPPORTED",
    "HGptiResult": "HGPTI_ERROR_NOT_SUPPORTED",
}


def make_api(return_type, name="unsupported_api", parameters=None):
    return {
        "name": name,
        "return_type": return_type,
        "parameters": parameters or [],
        "library": "test",
    }


class UnsupportedStubsTests(unittest.TestCase):
    def test_each_status_type_returns_not_supported(self):
        for return_type, status in EXPECTED_STATUSES.items():
            with self.subTest(return_type=return_type):
                stub = generator._generate_stub_function(make_api(return_type))
                self.assertIn(f"static inline {return_type} unsupported_api(void)", stub)
                self.assertIn('fprintf(stderr, "unsupported_api is not supported.\\n");', stub)
                self.assertIn(f"return {status};", stub)
                self.assertNotIn("exit(", stub)
                self.assertNotIn("unreachable", stub)

    def test_return_type_whitespace_is_normalized(self):
        stub = generator._generate_stub_function(make_api(" acblasStatus_t \t"))
        self.assertIn("return ACBLAS_STATUS_NOT_SUPPORTED;", stub)
        self.assertNotIn("exit(", stub)

    def test_parameter_formatting_and_unused_casts(self):
        api = make_api("acsolverStatus_t", parameters=[
            {"name": "buffer", "type": "int * {}"},
            {"name": "count", "type": "size_t"},
            {"name": "values", "type": "float {}[4]"},
        ])
        stub = generator._generate_stub_function(api)
        self.assertIn("int * buffer, size_t count, float values[4]", stub)
        self.assertIn("(void)buffer; (void)count; (void)values;", stub)
        self.assertIn("return ACSOLVER_STATUS_NOT_SUPPORTED;", stub)

    def test_types_without_not_supported_keep_fatal_behavior(self):
        for return_type in ("void", "const char *", "void *", "unsigned int",
                            "size_t", "int", "long", "acrandStatus_t", "hgrtcResult"):
            with self.subTest(return_type=return_type):
                stub = generator._generate_stub_function(make_api(return_type))
                self.assertIn("exit(1);", stub)
                if return_type == "void":
                    self.assertNotIn("return ", stub)
                else:
                    self.assertIn(f"return ({return_type})0;  /* unreachable */", stub)

    def test_all_configured_apis_are_generated_with_correct_behavior(self):
        config_path = Path(generator.__file__).with_name("unsupported_apis.json")
        with config_path.open() as config_file:
            apis = json.load(config_file)["apis"]
        header = generator.generate_stubs_header()
        self.assertEqual(len(apis), 1022)
        self.assertEqual(header.count("static inline "), len(apis))
        returning = 0
        for api in apis:
            with self.subTest(api=api["name"]):
                stub = generator._generate_stub_function(api)
                self.assertIn(stub, header)
                status = EXPECTED_STATUSES.get(api["return_type"].strip())
                if status is not None:
                    returning += 1
                    self.assertIn(f"return {status};", stub)
                    self.assertNotIn("exit(", stub)
                else:
                    self.assertIn("exit(1);", stub)
        self.assertEqual(returning, 998)
        self.assertEqual(header.count("exit(1);"), 24)

    def test_write_stubs_header_uses_updated_generator(self):
        with tempfile.TemporaryDirectory() as output_dir:
            header_path = generator.write_stubs_header(output_dir)
            self.assertEqual(Path(header_path).name, "ppu_unsupported_stubs.h")
            self.assertEqual(Path(header_path).read_text(), generator.generate_stubs_header())

    def test_generated_status_stubs_compile_and_return_in_c_and_cpp(self):
        # Minimal SDK stand-ins test syntax and runtime returns, not real SDK compatibility.
        compilers = [(shutil.which("cc"), "c", "c99"),
                     (shutil.which("c++"), "cpp", "c++11")]
        if not any(compiler for compiler, _, _ in compilers):
            self.skipTest("No C/C++ compiler installed")
        source = ["#include <stdio.h>", "#include <stdlib.h>"]
        calls = []
        for index, (return_type, status) in enumerate(EXPECTED_STATUSES.items(), 1):
            name = f"unsupported_status_{index}"
            source.append(f"typedef enum {{ {status} = {index} }} {return_type};")
            source.append(generator._generate_stub_function(make_api(
                return_type, name=name,
                parameters=[{"name": "output", "type": "int * {}"}],
            )))
            calls.append(f"    if ({name}(&output) != {status}) return {index};")
            calls.append("    if (output != 123) return 99;")
        source.extend(["int main(void) {", "    int output = 123;", *calls,
                       "    return 0;", "}"])
        with tempfile.TemporaryDirectory() as build_dir:
            for compiler, extension, standard in compilers:
                if compiler is None:
                    continue
                with self.subTest(language=extension):
                    source_path = Path(build_dir) / f"stubs.{extension}"
                    executable = Path(build_dir) / f"stubs_{extension}"
                    source_path.write_text("\n".join(source))
                    build = subprocess.run(
                        [compiler, f"-std={standard}", "-Wall", "-Wextra", "-Werror",
                         str(source_path), "-o", str(executable)],
                        capture_output=True, text=True, timeout=30,
                    )
                    self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
                    run = subprocess.run([str(executable)], capture_output=True,
                                         text=True, timeout=10)
                    self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                    self.assertEqual(run.stderr.count("is not supported."),
                                     len(EXPECTED_STATUSES))


if __name__ == "__main__":
    unittest.main()
