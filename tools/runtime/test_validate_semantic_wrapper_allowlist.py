#!/usr/bin/env python3
# ===----------------------------------------------------------------------===//
#
# Part of the OpenCyphal project, under the MIT licence
# SPDX-License-Identifier: MIT
#
# ===----------------------------------------------------------------------===//

"""Regression tests for runtime semantic-wrapper allowlist validation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


class AllowlistValidatorRegressionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(self.repo_root_arg).resolve()
        self.validator = self.repo_root / "tools/runtime/validate_semantic_wrapper_allowlist.py"
        self.allowlist = self.repo_root / "runtime/semantic_wrapper_allowlist.json"
        self.assertTrue(self.validator.exists(), f"missing validator script: {self.validator}")
        self.assertTrue(self.allowlist.exists(), f"missing allowlist file: {self.allowlist}")

    def _run_validator(self, allowlist_path: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(self.validator),
                "--repo-root",
                str(self.repo_root),
                "--allowlist",
                str(allowlist_path),
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def _append_entry(self, data: Dict[str, object], entry: Dict[str, object]) -> Dict[str, object]:
        entries = data["entries"]
        assert isinstance(entries, list)
        return {"version": data["version"], "entries": [*entries, entry]}

    def test_baseline_allowlist_passes(self) -> None:
        result = self._run_validator(self.allowlist)
        self.assertEqual(
            result.returncode,
            0,
            msg=f"expected validator success, got rc={result.returncode}, stderr={result.stderr}",
        )

    def test_generated_python_loader_entry_fails(self) -> None:
        baseline = _load_json(self.allowlist)
        mutated = self._append_entry(
            baseline,
            {
                "language": "python",
                "file": "runtime/python/_runtime_loader.py",
                "symbol": "BACKEND",
                "kind": "semantic_wrapper",
                "owner": "@llvmdsdl-runtime",
                "rationale": "stale exception entry regression test coverage for generated runtime file",
            },
        )
        with tempfile.TemporaryDirectory(prefix="llvmdsdl-allowlist-test-") as tmp_dir:
            tmp_allowlist = Path(tmp_dir) / "allowlist.json"
            _write_json(tmp_allowlist, mutated)
            result = self._run_validator(tmp_allowlist)
        self.assertNotEqual(result.returncode, 0, msg="expected validator to fail for generated-file allowlist entry")
        self.assertIn("generated file", result.stderr)
        self.assertIn("runtime/python/_runtime_loader.py", result.stderr)

    def test_generated_rust_wrapper_entry_fails(self) -> None:
        baseline = _load_json(self.allowlist)
        mutated = self._append_entry(
            baseline,
            {
                "language": "rust",
                "file": "runtime/rust/dsdl_runtime_semantic_wrappers.rs",
                "symbol": "DsdlVec",
                "kind": "semantic_wrapper",
                "owner": "@llvmdsdl-runtime",
                "rationale": "stale exception entry regression test coverage for generated runtime file",
            },
        )
        with tempfile.TemporaryDirectory(prefix="llvmdsdl-allowlist-test-") as tmp_dir:
            tmp_allowlist = Path(tmp_dir) / "allowlist.json"
            _write_json(tmp_allowlist, mutated)
            result = self._run_validator(tmp_allowlist)
        self.assertNotEqual(result.returncode, 0, msg="expected validator to fail for generated-file allowlist entry")
        self.assertIn("generated file", result.stderr)
        self.assertIn("runtime/rust/dsdl_runtime_semantic_wrappers.rs", result.stderr)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run allowlist validator regression tests.")
    parser.add_argument("--repo-root", required=True, help="Path to repository root.")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    AllowlistValidatorRegressionTest.repo_root_arg = args.repo_root
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AllowlistValidatorRegressionTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
