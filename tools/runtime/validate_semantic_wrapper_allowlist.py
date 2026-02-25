#!/usr/bin/env python3
# ===----------------------------------------------------------------------===//
#
# Part of the OpenCyphal project, under the MIT licence
# SPDX-License-Identifier: MIT
#
# ===----------------------------------------------------------------------===//

"""Validate runtime semantic-wrapper exception allowlist integrity."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


SUPPORTED_LANGUAGES = {"c", "cpp", "rust", "go", "ts", "python", "python_accel"}
REQUIRED_FIELDS = ("language", "file", "symbol", "kind", "owner", "rationale")
DEFAULT_ALLOWLIST = "runtime/semantic_wrapper_allowlist.json"
GENERATED_FILE_MARKER = "LLVMDSDL AUTO-GENERATED FILE"


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_generated_runtime_file(text: str) -> bool:
    return GENERATED_FILE_MARKER in text


def _find_generated_wrapper_symbols(runtime_rust_text: str) -> List[str]:
    symbols: List[str] = []
    lines = runtime_rust_text.splitlines()
    marker = re.compile(r"wrapper used by generated", re.IGNORECASE)
    symbol = re.compile(r"pub\s+(?:struct|type)\s+([A-Za-z_][A-Za-z0-9_]*)")
    for idx, line in enumerate(lines):
        if not marker.search(line):
            continue
        window = lines[idx : idx + 8]
        joined = "\n".join(window)
        match = symbol.search(joined)
        if match:
            symbols.append(match.group(1))
    return sorted(set(symbols))


def _find_python_runtime_wrapper_symbols(runtime_loader_text: str) -> List[str]:
    symbols: set[str] = set()
    for match in re.finditer(r"['\"](LLVMDSDL_[A-Z0-9_]+)['\"]", runtime_loader_text):
        symbols.add(match.group(1))
    if re.search(r"^BACKEND\s*=", runtime_loader_text, flags=re.MULTILINE):
        symbols.add("BACKEND")
    return sorted(symbols)


def _find_go_runtime_wrapper_symbols(runtime_go_text: str) -> List[str]:
    symbols: set[str] = set()
    type_decl = re.compile(r"^type\s+([A-Za-z_][A-Za-z0-9_]*)\s+(?:struct|interface)\b", flags=re.MULTILINE)
    semantic_name = re.compile(r"(Provider|Contract|Mode|Vec|Array|Wrapper)$")
    for match in type_decl.finditer(runtime_go_text):
        symbol = match.group(1)
        if semantic_name.search(symbol):
            symbols.add(symbol)
    return sorted(symbols)


def _validate_entries(repo_root: Path, entries: Sequence[Dict[str, object]]) -> List[str]:
    failures: List[str] = []
    seen: set[Tuple[str, str, str]] = set()

    for idx, entry in enumerate(entries):
        where = f"entries[{idx}]"
        missing = [field for field in REQUIRED_FIELDS if field not in entry]
        if missing:
            failures.append(f"{where} missing required fields: {', '.join(missing)}")
            continue

        language = str(entry["language"])
        file_rel = str(entry["file"])
        symbol = str(entry["symbol"])
        kind = str(entry["kind"])
        owner = str(entry["owner"])
        rationale = str(entry["rationale"])

        if language not in SUPPORTED_LANGUAGES:
            failures.append(f"{where} has unsupported language '{language}'")
        if kind != "semantic_wrapper":
            failures.append(f"{where} has unsupported kind '{kind}' (expected semantic_wrapper)")
        if not owner.startswith("@") or len(owner) < 2:
            failures.append(f"{where} has invalid owner '{owner}' (expected @owner)")
        if len(rationale.strip()) < 24:
            failures.append(f"{where} rationale too short; include concrete owner-facing rationale")

        key = (language, file_rel, symbol)
        if key in seen:
            failures.append(f"{where} duplicates entry ({language}, {file_rel}, {symbol})")
        seen.add(key)

        file_path = repo_root / file_rel
        if not file_path.exists():
            failures.append(f"{where} references missing file: {file_rel}")
            continue
        text = file_path.read_text(encoding="utf-8")
        if _is_generated_runtime_file(text):
            failures.append(
                f"{where} references generated file '{file_rel}'; generated semantic wrappers must not be allowlisted"
            )
            continue
        if symbol not in text:
            failures.append(f"{where} symbol '{symbol}' not found in {file_rel}")

    return failures


def _validate_generated_wrapper_coverage(repo_root: Path, entries: Sequence[Dict[str, object]]) -> List[str]:
    failures: List[str] = []
    rust_path = repo_root / "runtime/rust/dsdl_runtime_semantic_wrappers.rs"
    if not rust_path.exists():
        return failures

    rust_text = rust_path.read_text(encoding="utf-8")
    if _is_generated_runtime_file(rust_text):
        return failures

    discovered = _find_generated_wrapper_symbols(rust_text)
    allowlisted = {
        str(entry.get("symbol", ""))
        for entry in entries
        if str(entry.get("file", "")) == "runtime/rust/dsdl_runtime_semantic_wrappers.rs"
    }

    for symbol in discovered:
        if symbol not in allowlisted:
            failures.append(
                "generated-wrapper coverage gap: runtime/rust/dsdl_runtime_semantic_wrappers.rs symbol "
                f"'{symbol}' is not listed in runtime/semantic_wrapper_allowlist.json"
            )
    return failures


def _validate_python_wrapper_coverage(repo_root: Path, entries: Sequence[Dict[str, object]]) -> List[str]:
    failures: List[str] = []
    py_loader_path = repo_root / "runtime/python/_runtime_loader.py"
    if not py_loader_path.exists():
        return failures

    py_text = py_loader_path.read_text(encoding="utf-8")
    if _is_generated_runtime_file(py_text):
        return failures

    discovered = _find_python_runtime_wrapper_symbols(py_text)
    allowlisted = {
        str(entry.get("symbol", ""))
        for entry in entries
        if str(entry.get("file", "")) == "runtime/python/_runtime_loader.py"
    }

    for symbol in discovered:
        if symbol not in allowlisted:
            failures.append(
                "python runtime-wrapper coverage gap: runtime/python/_runtime_loader.py symbol "
                f"'{symbol}' is not listed in runtime/semantic_wrapper_allowlist.json"
            )
    return failures


def _validate_go_wrapper_coverage(repo_root: Path, entries: Sequence[Dict[str, object]]) -> List[str]:
    failures: List[str] = []
    go_runtime_path = repo_root / "runtime/go/dsdl_runtime.go"
    if not go_runtime_path.exists():
        return failures

    go_text = go_runtime_path.read_text(encoding="utf-8")
    discovered = _find_go_runtime_wrapper_symbols(go_text)
    allowlisted = {
        str(entry.get("symbol", ""))
        for entry in entries
        if str(entry.get("file", "")) == "runtime/go/dsdl_runtime.go"
    }

    for symbol in discovered:
        if symbol not in allowlisted:
            failures.append(
                "go runtime-wrapper coverage gap: runtime/go/dsdl_runtime.go symbol "
                f"'{symbol}' is not listed in runtime/semantic_wrapper_allowlist.json"
            )
    return failures


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate runtime semantic-wrapper exception allowlist.")
    parser.add_argument("--repo-root", required=True, help="Path to repository root.")
    parser.add_argument(
        "--allowlist",
        default=DEFAULT_ALLOWLIST,
        help=f"Path to allowlist JSON relative to repo root (default: {DEFAULT_ALLOWLIST}).",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    allowlist_path = (repo_root / args.allowlist).resolve()
    if not allowlist_path.exists():
        print(f"runtime allowlist regression: allowlist not found: {allowlist_path}", file=sys.stderr)
        return 2

    raw = _load_json(allowlist_path)
    if int(raw.get("version", 0)) != 1:
        print("runtime allowlist regression: expected allowlist version == 1", file=sys.stderr)
        return 2

    entries_raw = raw.get("entries")
    if not isinstance(entries_raw, list):
        print("runtime allowlist regression: expected 'entries' array", file=sys.stderr)
        return 2

    entries: List[Dict[str, object]] = []
    for idx, entry in enumerate(entries_raw):
        if not isinstance(entry, dict):
            print(f"runtime allowlist regression: entries[{idx}] must be an object", file=sys.stderr)
            return 2
        entries.append(entry)

    failures: List[str] = []
    failures.extend(_validate_entries(repo_root, entries))
    failures.extend(_validate_generated_wrapper_coverage(repo_root, entries))
    failures.extend(_validate_python_wrapper_coverage(repo_root, entries))
    failures.extend(_validate_go_wrapper_coverage(repo_root, entries))
    if failures:
        for failure in failures:
            print(f"runtime allowlist regression: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
