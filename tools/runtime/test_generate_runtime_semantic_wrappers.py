#!/usr/bin/env python3
# ===----------------------------------------------------------------------===//
#
# Part of the OpenCyphal project, under the MIT licence
# SPDX-License-Identifier: MIT
#
# ===----------------------------------------------------------------------===//

"""Regression tests for runtime semantic-wrapper generation tooling."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List


class RuntimeSemanticWrapperGenerationRegressionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(self.repo_root_arg).resolve()
        self.generator = self.repo_root / "tools/runtime/generate_runtime_semantic_wrappers.py"
        self.assertTrue(self.generator.exists(), f"missing generator script: {self.generator}")

    def _run_generator(self, repo_root: Path, check: bool) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, str(self.generator), "--repo-root", str(repo_root)]
        if check:
            command.append("--check")
        return subprocess.run(command, text=True, capture_output=True, check=False)

    def _make_temp_repo_fixture(self) -> Path:
        tmp_root = Path(tempfile.mkdtemp(prefix="llvmdsdl-runtime-gen-test-"))
        (tmp_root / "runtime/rust").mkdir(parents=True, exist_ok=True)
        (tmp_root / "runtime/python").mkdir(parents=True, exist_ok=True)

        fixtures = [
            "runtime/rust/dsdl_runtime_semantic_wrappers.rs.in",
            "runtime/rust/dsdl_runtime_semantic_wrappers.rs",
            "runtime/python/_runtime_loader.py.in",
            "runtime/python/_runtime_loader.py",
        ]
        for rel in fixtures:
            src = self.repo_root / rel
            dst = tmp_root / rel
            shutil.copy2(src, dst)
        return tmp_root

    def test_generation_check_passes_for_baseline_outputs(self) -> None:
        temp_repo = self._make_temp_repo_fixture()
        result = self._run_generator(temp_repo, check=True)
        self.assertEqual(
            result.returncode,
            0,
            msg=f"expected generation check success, got rc={result.returncode}, stdout={result.stdout}, stderr={result.stderr}",
        )

    def test_generation_drift_is_detected_and_repaired(self) -> None:
        temp_repo = self._make_temp_repo_fixture()
        rust_output = temp_repo / "runtime/rust/dsdl_runtime_semantic_wrappers.rs"
        rust_output.write_text(rust_output.read_text(encoding="utf-8") + "\n// drift\n", encoding="utf-8")

        drift = self._run_generator(temp_repo, check=True)
        self.assertNotEqual(drift.returncode, 0, msg="expected generation check to fail on drift")
        self.assertIn("out of date", drift.stdout)

        repair = self._run_generator(temp_repo, check=False)
        self.assertEqual(repair.returncode, 0, msg=f"expected repair success, stderr={repair.stderr}")

        post = self._run_generator(temp_repo, check=True)
        self.assertEqual(post.returncode, 0, msg=f"expected post-repair check success, stderr={post.stderr}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run runtime semantic-wrapper generation regression tests.")
    parser.add_argument("--repo-root", required=True, help="Path to repository root.")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    RuntimeSemanticWrapperGenerationRegressionTest.repo_root_arg = args.repo_root
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeSemanticWrapperGenerationRegressionTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
