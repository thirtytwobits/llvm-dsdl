#!/usr/bin/env python3
# ===----------------------------------------------------------------------===//
#
# Part of the OpenCyphal project, under the MIT licence
# SPDX-License-Identifier: MIT
#
# ===----------------------------------------------------------------------===//

"""Generate malformed-input contract matrix coverage from configured CTest lanes."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

CATEGORIES: List[Tuple[str, str]] = [
    ("out_of_range_bit_ops", "Out-of-range bit read/copy handling"),
    ("copy_range_violations", "Bit-copy range violation handling"),
    ("invalid_union_tag", "Invalid union-tag handling"),
    ("invalid_delimiter_header", "Invalid delimited-header handling"),
    ("capacity_precheck", "Section serialization-capacity precheck"),
]

MATRIX: Dict[str, Dict[str, object]] = {
    "ts_portable_fast": {
        "behavior": "Error-throwing runtime paths validated by TS smoke/parity lanes.",
        "cells": {
            "out_of_range_bit_ops": [r"^llvmdsdl-ts-runtime-truncated-decode-smoke$", r"^llvmdsdl-fixtures-c-ts-truncated-decode-parity$"],
            "copy_range_violations": [r"^llvmdsdl-ts-runtime-truncated-decode-smoke$", r"^llvmdsdl-fixtures-c-ts-truncated-decode-parity$"],
            "invalid_union_tag": [r"^llvmdsdl-fixtures-c-ts-union-parity$"],
            "invalid_delimiter_header": [r"^llvmdsdl-ts-runtime-delimited-invalid-header-smoke$", r"^llvmdsdl-fixtures-c-ts-delimited-invalid-header-parity$"],
            "capacity_precheck": [r"^llvmdsdl-fixtures-c-ts-runtime-parity$"],
        },
    },
    "python_portable_fast_accel": {
        "behavior": "Mode-specific malformed contract validated (`portable|fast|accel`) with fuzz/parity lanes.",
        "cells": {
            "out_of_range_bit_ops": [r"^llvmdsdl-fixtures-python-malformed-input-contract$", r"^llvmdsdl-fixtures-python-malformed-decode-fuzz-parity$"],
            "copy_range_violations": [r"^llvmdsdl-fixtures-python-malformed-input-contract$", r"^llvmdsdl-fixtures-python-malformed-decode-fuzz-parity$"],
            "invalid_union_tag": [r"^llvmdsdl-fixtures-c-python-union-parity$"],
            "invalid_delimiter_header": [r"^llvmdsdl-fixtures-c-python-delimited-parity$"],
            "capacity_precheck": [r"^llvmdsdl-fixtures-python-runtime-parity$"],
        },
    },
    "rust_portable_fast": {
        "behavior": "Rust runtime contract unit tests plus cross-language parity lanes enforce malformed behavior classes.",
        "cells": {
            "out_of_range_bit_ops": [r"^llvmdsdl-fixtures-rust-runtime-unit-tests$", r"^llvmdsdl-fixtures-rust-runtime-unit-tests-no-std-pool$"],
            "copy_range_violations": [r"^llvmdsdl-fixtures-rust-runtime-unit-tests$", r"^llvmdsdl-fixtures-rust-runtime-unit-tests-no-std-pool$"],
            "invalid_union_tag": [r"^llvmdsdl-uavcan-c-rust-parity$"],
            "invalid_delimiter_header": [r"^llvmdsdl-uavcan-c-rust-parity$"],
            "capacity_precheck": [r"^llvmdsdl-fixtures-rust-runtime-unit-tests$"],
        },
    },
    "native_c_cpp_go_boundaries": {
        "behavior": "Native C/C++/Go parity lanes enforce malformed decode/capacity behavior compatibility.",
        "cells": {
            "out_of_range_bit_ops": [r"^llvmdsdl-uavcan-cpp-c-parity$", r"^llvmdsdl-uavcan-c-go-parity$"],
            "copy_range_violations": [r"^llvmdsdl-uavcan-cpp-c-parity$", r"^llvmdsdl-uavcan-c-go-parity$"],
            "invalid_union_tag": [r"^llvmdsdl-uavcan-cpp-c-parity$", r"^llvmdsdl-uavcan-c-go-parity$"],
            "invalid_delimiter_header": [r"^llvmdsdl-uavcan-cpp-c-parity$", r"^llvmdsdl-uavcan-c-go-parity$"],
            "capacity_precheck": [r"^llvmdsdl-uavcan-cpp-c-parity$", r"^llvmdsdl-uavcan-c-go-parity$"],
        },
    },
}


NATIVE_BEHAVIOR_GUARDS: Dict[str, Dict[str, object]] = {
    "cpp_c_directed_markers": {
        "file": "test/integration/RunCppCParity.cmake",
        "marker_prefix": "INFO cpp-c directed marker ",
    },
    "c_rust_directed_markers": {
        "file": "test/integration/RunCRustParity.cmake",
        "marker_prefix": "INFO c/rust directed marker ",
    },
    "c_go_directed_coverage_summary": {
        "file": "test/integration/RunCGoParity.cmake",
        "required_patterns": [
            r"PASS directed coverage any=\(\[0-9\]\+\) truncation=\(\[0-9\]\+\) serialize_buffer=\(\[0-9\]\+\)",
            r"if\(NOT coverage_any EQUAL observed_cases",
            r"NOT coverage_truncation EQUAL observed_cases",
            r"NOT coverage_serialize_buffer EQUAL observed_cases",
        ],
    },
}


def _extract_markers_from_script(script_text: str, marker_prefix: str) -> Set[str]:
    escaped_prefix = re.escape(marker_prefix)
    marker_pattern = re.compile(rf'"{escaped_prefix}([^"]+)"')
    return {match.group(1).strip() for match in marker_pattern.finditer(script_text)}


def _match_any_marker(markers: Set[str], marker_regexes: Iterable[str]) -> bool:
    compiled = [re.compile(pattern) for pattern in marker_regexes]
    return any(any(pattern.search(marker) for pattern in compiled) for marker in sorted(markers))


def _build_native_behavior_guards(repo_root: Path) -> Dict[str, object]:
    out: Dict[str, object] = {"checks": {}, "all_passed": True}
    category_marker_patterns: Dict[str, List[str]] = {
        "out_of_range_bit_ops": [r"truncated_payload_roundtrip", r"heartbeat_empty_deserialize"],
        "copy_range_violations": [r"truncated_payload_roundtrip"],
        "invalid_union_tag": [r"union_tag"],
        "invalid_delimiter_header": [r"delimiter_header"],
        "capacity_precheck": [r"too_small_serialize"],
    }

    cpp_markers: Set[str] = set()
    rust_markers: Set[str] = set()

    for guard_name, guard_cfg in NATIVE_BEHAVIOR_GUARDS.items():
        script_path = repo_root / str(guard_cfg["file"])
        if not script_path.is_file():
            out["checks"][guard_name] = {
                "pass": False,
                "reason": f"missing script: {script_path}",
            }
            out["all_passed"] = False
            continue

        script_text = script_path.read_text(encoding="utf-8")
        if "marker_prefix" in guard_cfg:
            marker_prefix = str(guard_cfg["marker_prefix"])
            markers = _extract_markers_from_script(script_text, marker_prefix)
            category_results: Dict[str, bool] = {}
            missing_categories: List[str] = []
            for category_id in category_marker_patterns:
                ok = _match_any_marker(markers, category_marker_patterns[category_id])
                category_results[category_id] = ok
                if not ok:
                    missing_categories.append(category_id)
            check_pass = len(missing_categories) == 0
            out["checks"][guard_name] = {
                "pass": check_pass,
                "script": str(script_path.relative_to(repo_root)),
                "marker_count": len(markers),
                "category_results": category_results,
                "missing_categories": missing_categories,
            }
            if guard_name == "cpp_c_directed_markers":
                cpp_markers = markers
            if guard_name == "c_rust_directed_markers":
                rust_markers = markers
            if not check_pass:
                out["all_passed"] = False
            continue

        required_patterns = [re.compile(pattern) for pattern in guard_cfg["required_patterns"]]
        missing_patterns = [pattern.pattern for pattern in required_patterns if not pattern.search(script_text)]
        check_pass = len(missing_patterns) == 0
        out["checks"][guard_name] = {
            "pass": check_pass,
            "script": str(script_path.relative_to(repo_root)),
            "missing_patterns": missing_patterns,
        }
        if not check_pass:
            out["all_passed"] = False

    if cpp_markers and rust_markers:
        missing_in_rust = sorted(cpp_markers - rust_markers)
        extra_in_rust = sorted(rust_markers - cpp_markers)
        aligned = len(missing_in_rust) == 0 and len(extra_in_rust) == 0
        out["checks"]["cpp_c_vs_c_rust_marker_alignment"] = {
            "pass": aligned,
            "missing_in_c_rust": missing_in_rust,
            "extra_in_c_rust": extra_in_rust,
        }
        if not aligned:
            out["all_passed"] = False
    else:
        out["checks"]["cpp_c_vs_c_rust_marker_alignment"] = {
            "pass": False,
            "reason": "missing marker sets for cpp-c or c-rust guard checks",
        }
        out["all_passed"] = False

    return out


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


def _build_report(repo_root: Path, test_dir: Path, config: str | None) -> Dict[str, object]:
    test_names = _extract_test_names_from_ctest(test_dir, config)
    category_labels = {category_id: category_label for category_id, category_label in CATEGORIES}

    backends: Dict[str, object] = {}
    missing_cells: List[Dict[str, str]] = []
    total_cells = 0
    covered_cells = 0

    for backend in sorted(MATRIX):
        backend_cfg = MATRIX[backend]
        cells = backend_cfg["cells"]
        row_cells: Dict[str, object] = {}
        backend_total = 0
        backend_covered = 0

        for category_id, _ in CATEGORIES:
            backend_total += 1
            total_cells += 1
            patterns = cells[category_id]
            evidence = _matches(test_names, patterns)
            covered = len(evidence) > 0
            if covered:
                backend_covered += 1
                covered_cells += 1
            else:
                missing_cells.append({"backend": backend, "category": category_id})

            row_cells[category_id] = {
                "label": category_labels[category_id],
                "covered": covered,
                "patterns": list(patterns),
                "evidence_tests": evidence,
            }

        backends[backend] = {
            "behavior": backend_cfg["behavior"],
            "covered_cells": backend_covered,
            "total_cells": backend_total,
            "score": int(round((backend_covered * 100.0) / backend_total)),
            "cells": row_cells,
        }

    try:
        test_dir_text = str(test_dir.relative_to(repo_root))
    except ValueError:
        test_dir_text = str(test_dir)

    native_behavior_guards = _build_native_behavior_guards(repo_root)

    return {
        "version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_source": f"ctest --test-dir {test_dir_text}",
        "test_name_count": len(test_names),
        "categories": [{"id": category_id, "label": category_label} for category_id, category_label in CATEGORIES],
        "backends": backends,
        "overall_covered_cells": covered_cells,
        "overall_total_cells": total_cells,
        "overall_score": int(round((covered_cells * 100.0) / total_cells)),
        "missing_cells": missing_cells,
        "native_behavior_guards": native_behavior_guards,
    }


def _write_json(path: Path, report: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_markdown(path: Path, report: Dict[str, object]) -> None:
    lines: List[str] = []
    lines.append("# Malformed-Input Contract Matrix")
    lines.append("")
    lines.append("Generated by `tools/convergence/malformed_contract_matrix_report.py`.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Backend mode family | Score | Covered / Total |")
    lines.append("| --- | ---: | ---: |")
    for backend in sorted(report["backends"]):
        row = report["backends"][backend]
        lines.append(f"| `{backend}` | `{row['score']}` | `{row['covered_cells']} / {row['total_cells']}` |")
    lines.append("")
    lines.append(f"Overall matrix score: `{report['overall_score']}`")
    lines.append(f"Integration test names scanned: `{report['test_name_count']}`")
    lines.append(f"Test-source scan: `{report['test_source']}`")
    lines.append("")

    for backend in sorted(report["backends"]):
        row = report["backends"][backend]
        lines.append(f"## `{backend}`")
        lines.append("")
        lines.append(row["behavior"])
        lines.append("")
        lines.append("| Category | Status | Evidence |")
        lines.append("| --- | --- | --- |")
        for category_id, category_label in CATEGORIES:
            cell = row["cells"][category_id]
            status = "covered" if cell["covered"] else "missing"
            evidence = ", ".join(f"`{name}`" for name in cell["evidence_tests"]) if cell["evidence_tests"] else "-"
            lines.append(f"| `{category_label}` | `{status}` | {evidence} |")
        lines.append("")

    if report["missing_cells"]:
        lines.append("## Missing Cells")
        lines.append("")
        for cell in report["missing_cells"]:
            lines.append(f"- backend `{cell['backend']}` category `{cell['category']}`")
        lines.append("")
    else:
        lines.append("## Missing Cells")
        lines.append("")
        lines.append("None.")
        lines.append("")

    lines.append("## Native Behavior Guards")
    lines.append("")
    behavior = report.get("native_behavior_guards", {})
    lines.append(f"All checks passed: `{bool(behavior.get('all_passed', False))}`")
    lines.append("")
    lines.append("| Guard | Status | Notes |")
    lines.append("| --- | --- | --- |")
    for guard_name in sorted(behavior.get("checks", {})):
        guard = behavior["checks"][guard_name]
        status = "pass" if guard.get("pass", False) else "fail"
        notes = []
        if "script" in guard:
            notes.append(f"script `{guard['script']}`")
        if "missing_categories" in guard and guard["missing_categories"]:
            notes.append("missing categories: " + ", ".join(f"`{x}`" for x in guard["missing_categories"]))
        if "missing_patterns" in guard and guard["missing_patterns"]:
            notes.append("missing patterns: " + ", ".join(f"`{x}`" for x in guard["missing_patterns"]))
        if "missing_in_c_rust" in guard and guard["missing_in_c_rust"]:
            notes.append("missing in c-rust: " + ", ".join(f"`{x}`" for x in guard["missing_in_c_rust"]))
        if "extra_in_c_rust" in guard and guard["extra_in_c_rust"]:
            notes.append("extra in c-rust: " + ", ".join(f"`{x}`" for x in guard["extra_in_c_rust"]))
        if "reason" in guard:
            notes.append(str(guard["reason"]))
        lines.append(f"| `{guard_name}` | `{status}` | {'; '.join(notes) if notes else '-'} |")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate malformed-input contract matrix coverage.")
    parser.add_argument("--repo-root", required=True, help="Repository root.")
    parser.add_argument("--ctest-test-dir", required=True, help="CTest build directory.")
    parser.add_argument("--ctest-config", help="CTest configuration (for multi-config generators).")
    parser.add_argument("--output-json", help="Output JSON path.")
    parser.add_argument("--output-md", help="Output Markdown path.")
    parser.add_argument("--check-regressions", action="store_true", help="Fail if any matrix cell is uncovered.")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    test_dir = Path(args.ctest_test_dir).resolve()

    try:
        report = _build_report(repo_root, test_dir, args.ctest_config)
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
                f"malformed-contract regression: backend '{cell['backend']}' category '{cell['category']}' has no coverage evidence",
                file=sys.stderr,
            )
        return 1
    if args.check_regressions and not report.get("native_behavior_guards", {}).get("all_passed", False):
        for guard_name, guard in sorted(report.get("native_behavior_guards", {}).get("checks", {}).items()):
            if guard.get("pass", False):
                continue
            print(f"malformed-contract native-behavior guard failed: '{guard_name}'", file=sys.stderr)
            if "missing_categories" in guard and guard["missing_categories"]:
                print("  missing categories: " + ", ".join(guard["missing_categories"]), file=sys.stderr)
            if "missing_patterns" in guard and guard["missing_patterns"]:
                print("  missing patterns: " + ", ".join(guard["missing_patterns"]), file=sys.stderr)
            if "missing_in_c_rust" in guard and guard["missing_in_c_rust"]:
                print("  missing in c-rust: " + ", ".join(guard["missing_in_c_rust"]), file=sys.stderr)
            if "extra_in_c_rust" in guard and guard["extra_in_c_rust"]:
                print("  extra in c-rust: " + ", ".join(guard["extra_in_c_rust"]), file=sys.stderr)
            if "reason" in guard:
                print("  reason: " + str(guard["reason"]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
