#!/usr/bin/env python3
# ===----------------------------------------------------------------------===//
#
# Part of the OpenCyphal project, under the MIT licence
# SPDX-License-Identifier: MIT
#
# ===----------------------------------------------------------------------===//

"""Generate and gate backend parity-matrix coverage from integration test lanes."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

FAMILIES: List[Tuple[str, str]] = [
    ("scalar", "Scalar"),
    ("fixed_array", "Fixed array"),
    ("variable_array", "Variable array"),
    ("union", "Union"),
    ("delimited", "Delimited"),
    ("service", "Service"),
    ("truncated_decode", "Truncated decode"),
    ("padding_alignment", "Padding/alignment"),
]

MATRIX: Dict[str, Dict[str, Dict[str, object]]] = {
    "c": {
        "scalar": {
            "mode": "fixture",
            "patterns": [r"^llvmdsdl-signed-narrow-c-ts-parity", r"^llvmdsdl-signed-narrow-c-python-parity"],
        },
        "fixed_array": {
            "mode": "fixture",
            "patterns": [r"^llvmdsdl-fixtures-c-python-fixed-array-parity", r"^llvmdsdl-fixtures-c-ts-composite-fixed-array-parity$"],
        },
        "variable_array": {
            "mode": "fixture",
            "patterns": [r"^llvmdsdl-fixtures-c-python-variable-array-parity", r"^llvmdsdl-fixtures-c-ts-variable-array-parity$"],
        },
        "union": {
            "mode": "fixture",
            "patterns": [r"^llvmdsdl-fixtures-c-python-union-parity", r"^llvmdsdl-fixtures-c-ts-union-parity$"],
        },
        "delimited": {
            "mode": "fixture",
            "patterns": [r"^llvmdsdl-fixtures-c-python-delimited-parity", r"^llvmdsdl-fixtures-c-ts-delimited-parity$"],
        },
        "service": {
            "mode": "fixture",
            "patterns": [r"^llvmdsdl-fixtures-c-python-service-parity", r"^llvmdsdl-fixtures-c-ts-service-parity$"],
        },
        "truncated_decode": {
            "mode": "fixture",
            "patterns": [r"^llvmdsdl-fixtures-c-python-truncated-decode-parity", r"^llvmdsdl-fixtures-c-ts-truncated-decode-parity$"],
        },
        "padding_alignment": {
            "mode": "fixture",
            "patterns": [r"^llvmdsdl-fixtures-c-python-padding-alignment-parity", r"^llvmdsdl-fixtures-c-ts-padding-alignment-parity$"],
        },
    },
    "cpp": {
        "scalar": {"mode": "fixture", "patterns": [r"^llvmdsdl-signed-narrow-cpp-c-parity"]},
        "fixed_array": {"mode": "corpus", "patterns": [r"^llvmdsdl-uavcan-cpp-c-parity"]},
        "variable_array": {"mode": "corpus", "patterns": [r"^llvmdsdl-uavcan-cpp-c-parity"]},
        "union": {"mode": "corpus", "patterns": [r"^llvmdsdl-uavcan-cpp-c-parity"]},
        "delimited": {"mode": "corpus", "patterns": [r"^llvmdsdl-uavcan-cpp-c-parity"]},
        "service": {"mode": "corpus", "patterns": [r"^llvmdsdl-uavcan-cpp-c-parity"]},
        "truncated_decode": {"mode": "corpus", "patterns": [r"^llvmdsdl-uavcan-cpp-c-parity"]},
        "padding_alignment": {"mode": "corpus", "patterns": [r"^llvmdsdl-uavcan-cpp-c-parity"]},
    },
    "rust": {
        "scalar": {"mode": "fixture", "patterns": [r"^llvmdsdl-signed-narrow-c-rust-parity"]},
        "fixed_array": {"mode": "corpus", "patterns": [r"^llvmdsdl-uavcan-c-rust-parity"]},
        "variable_array": {"mode": "corpus", "patterns": [r"^llvmdsdl-uavcan-c-rust-parity"]},
        "union": {"mode": "corpus", "patterns": [r"^llvmdsdl-uavcan-c-rust-parity"]},
        "delimited": {"mode": "corpus", "patterns": [r"^llvmdsdl-uavcan-c-rust-parity"]},
        "service": {"mode": "corpus", "patterns": [r"^llvmdsdl-uavcan-c-rust-parity"]},
        "truncated_decode": {"mode": "corpus", "patterns": [r"^llvmdsdl-uavcan-c-rust-parity"]},
        "padding_alignment": {"mode": "corpus", "patterns": [r"^llvmdsdl-uavcan-c-rust-parity"]},
    },
    "go": {
        "scalar": {"mode": "fixture", "patterns": [r"^llvmdsdl-signed-narrow-c-go-parity"]},
        "fixed_array": {"mode": "corpus", "patterns": [r"^llvmdsdl-uavcan-c-go-parity"]},
        "variable_array": {"mode": "corpus", "patterns": [r"^llvmdsdl-uavcan-c-go-parity"]},
        "union": {"mode": "corpus", "patterns": [r"^llvmdsdl-uavcan-c-go-parity"]},
        "delimited": {"mode": "corpus", "patterns": [r"^llvmdsdl-uavcan-c-go-parity"]},
        "service": {"mode": "corpus", "patterns": [r"^llvmdsdl-uavcan-c-go-parity"]},
        "truncated_decode": {"mode": "corpus", "patterns": [r"^llvmdsdl-uavcan-c-go-parity"]},
        "padding_alignment": {"mode": "corpus", "patterns": [r"^llvmdsdl-uavcan-c-go-parity"]},
    },
    "ts": {
        "scalar": {"mode": "fixture", "patterns": [r"^llvmdsdl-fixtures-c-ts-bigint-parity$", r"^llvmdsdl-fixtures-c-ts-float-parity$"]},
        "fixed_array": {"mode": "fixture", "patterns": [r"^llvmdsdl-fixtures-c-ts-composite-fixed-array-parity$"]},
        "variable_array": {"mode": "fixture", "patterns": [r"^llvmdsdl-fixtures-c-ts-variable-array-parity$", r"^llvmdsdl-fixtures-c-ts-composite-variable-array-parity$"]},
        "union": {"mode": "fixture", "patterns": [r"^llvmdsdl-fixtures-c-ts-union-parity$", r"^llvmdsdl-fixtures-c-ts-union-composite-parity$", r"^llvmdsdl-fixtures-c-ts-union-array-parity$"]},
        "delimited": {"mode": "fixture", "patterns": [r"^llvmdsdl-fixtures-c-ts-delimited-parity$", r"^llvmdsdl-fixtures-c-ts-delimited-invalid-header-parity$"]},
        "service": {"mode": "fixture", "patterns": [r"^llvmdsdl-fixtures-c-ts-service-parity$"]},
        "truncated_decode": {"mode": "fixture", "patterns": [r"^llvmdsdl-fixtures-c-ts-truncated-decode-parity$"]},
        "padding_alignment": {"mode": "fixture", "patterns": [r"^llvmdsdl-fixtures-c-ts-padding-alignment-parity$"]},
    },
    "python": {
        "scalar": {"mode": "fixture", "patterns": [r"^llvmdsdl-fixtures-c-python-bigint-parity", r"^llvmdsdl-fixtures-c-python-float-parity", r"^llvmdsdl-signed-narrow-c-python-parity"]},
        "fixed_array": {"mode": "fixture", "patterns": [r"^llvmdsdl-fixtures-c-python-fixed-array-parity"]},
        "variable_array": {"mode": "fixture", "patterns": [r"^llvmdsdl-fixtures-c-python-variable-array-parity"]},
        "union": {"mode": "fixture", "patterns": [r"^llvmdsdl-fixtures-c-python-union-parity"]},
        "delimited": {"mode": "fixture", "patterns": [r"^llvmdsdl-fixtures-c-python-delimited-parity"]},
        "service": {"mode": "fixture", "patterns": [r"^llvmdsdl-fixtures-c-python-service-parity"]},
        "truncated_decode": {"mode": "fixture", "patterns": [r"^llvmdsdl-fixtures-c-python-truncated-decode-parity"]},
        "padding_alignment": {"mode": "fixture", "patterns": [r"^llvmdsdl-fixtures-c-python-padding-alignment-parity"]},
    },
}


def _extract_test_names_from_integration_cmake(integration_cmake_text: str) -> Set[str]:
    name_pattern = re.compile(r"\bNAME\s+([A-Za-z0-9_.+-]+)")
    return set(name_pattern.findall(integration_cmake_text))


def _extract_test_names_from_ctest(test_dir: Path, config: str | None) -> Set[str]:
    cmd = ["ctest", "--test-dir", str(test_dir), "-N"]
    if config:
        cmd.extend(["-C", config])
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        stdout = proc.stdout.strip()
        detail = stderr if stderr else stdout
        raise RuntimeError(f"failed to list tests via ctest: {detail}")

    out: Set[str] = set()
    test_pattern = re.compile(r"^\s*Test\s+#\d+:\s+(.+?)\s*$")
    for line in proc.stdout.splitlines():
        match = test_pattern.match(line)
        if match:
            out.add(match.group(1))
    return out


def _matches(test_names: Iterable[str], patterns: Iterable[str]) -> List[str]:
    out: List[str] = []
    compiled = [re.compile(pattern) for pattern in patterns]
    for name in sorted(set(test_names)):
        if any(pattern.search(name) for pattern in compiled):
            out.append(name)
    return out


def _build_report(
    repo_root: Path, integration_cmake_path: Path, ctest_test_dir: Path | None, ctest_config: str | None
) -> Dict[str, object]:
    if ctest_test_dir is not None:
        test_names = _extract_test_names_from_ctest(ctest_test_dir, ctest_config)
        try:
            test_dir_text = str(ctest_test_dir.relative_to(repo_root))
        except ValueError:
            test_dir_text = str(ctest_test_dir)
        test_source = f"ctest --test-dir {test_dir_text}"
    else:
        test_names = _extract_test_names_from_integration_cmake(integration_cmake_path.read_text(encoding="utf-8"))
        test_source = str(integration_cmake_path.relative_to(repo_root))
    family_labels = {family_id: family_label for family_id, family_label in FAMILIES}

    backends: Dict[str, object] = {}
    missing_cells: List[Dict[str, str]] = []
    total_cells = 0
    covered_cells = 0

    for backend in sorted(MATRIX):
        backend_cells = MATRIX[backend]
        cells: Dict[str, object] = {}
        backend_total = 0
        backend_covered = 0

        for family_id, _ in FAMILIES:
            backend_total += 1
            total_cells += 1
            cell = backend_cells[family_id]
            matches = _matches(test_names, cell["patterns"])
            covered = len(matches) > 0
            if covered:
                backend_covered += 1
                covered_cells += 1
            else:
                missing_cells.append({"backend": backend, "family": family_id})

            cells[family_id] = {
                "label": family_labels[family_id],
                "mode": cell["mode"],
                "covered": covered,
                "evidence_tests": matches,
                "patterns": list(cell["patterns"]),
            }

        backends[backend] = {
            "covered_cells": backend_covered,
            "total_cells": backend_total,
            "score": int(round((backend_covered * 100.0) / backend_total)),
            "cells": cells,
        }

    return {
        "version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "integration_cmake": str(integration_cmake_path.relative_to(repo_root)),
        "test_source": test_source,
        "test_name_count": len(test_names),
        "families": [{"id": family_id, "label": family_label} for family_id, family_label in FAMILIES],
        "backends": backends,
        "overall_covered_cells": covered_cells,
        "overall_total_cells": total_cells,
        "overall_score": int(round((covered_cells * 100.0) / total_cells)),
        "missing_cells": missing_cells,
    }


def _write_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_markdown(path: Path, report: Dict[str, object]) -> None:
    lines: List[str] = []
    lines.append("# Parity Matrix Coverage")
    lines.append("")
    lines.append("Generated by `tools/convergence/parity_matrix_report.py`.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Backend | Score | Covered / Total |")
    lines.append("| --- | ---: | ---: |")
    for backend in sorted(report["backends"]):
        row = report["backends"][backend]
        lines.append(f"| `{backend}` | `{row['score']}` | `{row['covered_cells']} / {row['total_cells']}` |")
    lines.append("")
    lines.append(f"Overall matrix score: `{report['overall_score']}`")
    lines.append(f"Integration test names scanned: `{report['test_name_count']}`")
    lines.append(f"Test-source scan: `{report['test_source']}`")
    lines.append("")
    lines.append("`Mode` meanings:")
    lines.append("")
    lines.append("- `fixture`: category-specific lane(s) exist for this backend/family.")
    lines.append("- `corpus`: broad corpus parity lane used as coverage evidence for this family.")
    lines.append("")

    for backend in sorted(report["backends"]):
        row = report["backends"][backend]
        lines.append(f"## `{backend}`")
        lines.append("")
        lines.append("| Family | Mode | Status | Evidence |")
        lines.append("| --- | --- | --- | --- |")
        for family_id, family_label in FAMILIES:
            cell = row["cells"][family_id]
            status = "covered" if cell["covered"] else "missing"
            evidence = ", ".join(f"`{name}`" for name in cell["evidence_tests"]) if cell["evidence_tests"] else "-"
            lines.append(f"| `{family_label}` | `{cell['mode']}` | `{status}` | {evidence} |")
        lines.append("")

    if report["missing_cells"]:
        lines.append("## Missing Cells")
        lines.append("")
        for cell in report["missing_cells"]:
            lines.append(f"- backend `{cell['backend']}` family `{cell['family']}`")
        lines.append("")
    else:
        lines.append("## Missing Cells")
        lines.append("")
        lines.append("None.")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate parity-matrix coverage report for llvm-dsdl backends.")
    parser.add_argument("--repo-root", required=True, help="Repository root.")
    parser.add_argument(
        "--integration-cmake",
        default="test/integration/CMakeLists.txt",
        help="Integration CMakeLists path (relative to repo root unless absolute).",
    )
    parser.add_argument("--output-json", help="Output JSON path.")
    parser.add_argument("--output-md", help="Output Markdown path.")
    parser.add_argument("--ctest-test-dir", help="Configured CTest build directory used for dynamic test-name extraction.")
    parser.add_argument("--ctest-config", help="CTest configuration name (for multi-config generators).")
    parser.add_argument(
        "--check-regressions",
        action="store_true",
        help="Fail if any backend/family parity matrix cell is uncovered.",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    integration_cmake = Path(args.integration_cmake)
    if not integration_cmake.is_absolute():
        integration_cmake = repo_root / integration_cmake
    if not integration_cmake.exists():
        print(f"error: integration cmake not found: {integration_cmake}", file=sys.stderr)
        return 2

    ctest_test_dir = Path(args.ctest_test_dir).resolve() if args.ctest_test_dir else None
    try:
        report = _build_report(repo_root, integration_cmake, ctest_test_dir, args.ctest_config)
    except RuntimeError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if args.output_json:
        _write_json(Path(args.output_json), report)
    if args.output_md:
        _write_markdown(Path(args.output_md), report)

    if args.check_regressions and report["missing_cells"]:
        for cell in report["missing_cells"]:
            print(
                f"parity matrix regression: backend '{cell['backend']}' family '{cell['family']}' has no coverage evidence",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
