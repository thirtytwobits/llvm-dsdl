#!/usr/bin/env python3
# ===----------------------------------------------------------------------===//
#
# Part of the OpenCyphal project, under the MIT licence
# SPDX-License-Identifier: MIT
#
# ===----------------------------------------------------------------------===//

"""Compute MLIR convergence scorecards for llvm-dsdl backends."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SEMANTIC_CLASSES: List[Tuple[str, str]] = [
    ("field_step_ordering", "Field step ordering"),
    ("union_tag_mask_validate", "Union tag mask and validation"),
    ("scalar_normalize_sign_extend", "Scalar cast/normalize and sign extension"),
    ("variable_array_prefix_validate", "Variable array prefix normalize and length validation"),
    ("fixed_array_cardinality_validate", "Fixed array cardinality validation"),
    ("delimited_payload_validate", "Delimited payload header validation"),
    ("section_capacity_precheck", "Section capacity pre-check"),
    ("alignment_padding_orchestration", "Alignment/padding orchestration"),
    ("malformed_input_diagnostic_text", "Malformed-input diagnostic category text"),
    ("lowered_contract_validation", "Lowered contract schema/version validation"),
]

BACKEND_CONFIG = {
    "c": {
        "kind": "c",
        "path": "lib/CodeGen/CEmitter.cpp",
    },
    "cpp": {
        "kind": "native",
        "path": "lib/CodeGen/CppEmitter.cpp",
    },
    "rust": {
        "kind": "native",
        "path": "lib/CodeGen/RustEmitter.cpp",
    },
    "go": {
        "kind": "native",
        "path": "lib/CodeGen/GoEmitter.cpp",
    },
    "ts": {
        "kind": "scripted",
        "path": "lib/CodeGen/TsEmitter.cpp",
    },
    "python": {
        "kind": "scripted",
        "path": "lib/CodeGen/PythonEmitter.cpp",
    },
}


def _has_all(text: str, patterns: Iterable[str]) -> bool:
    return all(re.search(pattern, text, flags=re.MULTILINE) for pattern in patterns)


def _has_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.MULTILINE) for pattern in patterns)


def _load_text(repo_root: Path, rel_path: str) -> str:
    file_path = repo_root / rel_path
    return file_path.read_text(encoding="utf-8")


def _detect_traits(kind: str, text: str) -> Dict[str, bool]:
    traits: Dict[str, bool] = {}
    if kind == "c":
        traits["lower_pass"] = _has_any(text, [r"createLowerDSDLSerializationPass\("])
        traits["convert_pass"] = _has_any(text, [r"createConvertDSDLToEmitCPass\("])
        traits["schema_scan"] = _has_any(text, [r"dsdl\.schema", r"schemaByHeaderPath"])
        traits["schema_selection_guard"] = _has_any(text, [r"schema selection failed"])
        traits["emitc_pipeline"] = traits["lower_pass"] and traits["convert_pass"]
        return traits

    if kind == "native":
        traits["collect_lowered_facts"] = _has_any(text, [r"collectLoweredFactsFromMlir\("])
        traits["native_traversal"] = _has_any(text, [r"forEachNativeEmitterRenderStep\("])
        traits["native_function_skeleton"] = _has_any(text, [r"emitNativeFunctionSkeleton\("])
        traits["render_ir_helper_usage"] = _has_any(text, [r"renderIR\.helperBindings"])
        traits["helper_binding_render"] = _has_any(text, [r"renderSectionHelperBindings\("])
        traits["union_helper_usage"] = _has_any(text, [r"unionTagValidate"])
        traits["capacity_helper_usage"] = _has_any(text, [r"capacityCheck"])
        traits["shared_pipeline"] = traits["native_function_skeleton"] or _has_all(
            text,
            [
                r"collectLoweredFactsFromMlir\(",
                r"forEachNativeEmitterRenderStep\(",
                r"renderIR\.helperBindings",
                r"renderSectionHelperBindings\(",
                r"unionTagValidate",
                r"capacityCheck",
            ],
        )
        return traits

    if kind == "scripted":
        traits["collect_lowered_facts"] = _has_any(text, [r"collectLoweredFactsFromMlir\("])
        traits["runtime_plan"] = _has_any(text, [r"buildRuntimeSectionPlan\("])
        traits["scripted_body_plan"] = _has_any(
            text, [r"buildScriptedSectionBodyPlan\(", r"buildScriptedSectionOperationPlan\("]
        )
        traits["helper_plan_builder"] = _has_any(text, [r"buildSectionHelperBindingPlan\("])
        traits["helper_binding_render"] = _has_any(text, [r"renderSectionHelperBindings\("])
        traits["union_helper_usage"] = _has_any(text, [r"sectionHelperNames\.unionTagValidate"])
        traits["capacity_helper_usage"] = _has_any(text, [r"sectionHelperNames\.capacityCheck"])
        traits["scalar_helper_usage"] = _has_any(text, [r"helpers\.serScalar", r"helpers\.deserScalar"])
        traits["array_helper_usage"] = _has_any(
            text,
            [r"helpers\.arrayValidate", r"helpers\.serArrayPrefix", r"helpers\.deserArrayPrefix"],
        )
        traits["delimiter_helper_usage"] = _has_any(text, [r"helpers\.delimiterValidate"])
        traits["diagnostic_catalog"] = _has_any(text, [r"codegen_diagnostic_text::"])
        traits["shared_pipeline"] = _has_all(
            text,
            [
                r"collectLoweredFactsFromMlir\(",
                r"buildRuntimeSectionPlan\(",
                r"buildSectionHelperBindingPlan\(",
                r"renderSectionHelperBindings\(",
            ],
        ) and traits["scripted_body_plan"]
        return traits

    raise ValueError(f"unknown backend kind: {kind}")


def _classifications(kind: str, traits: Dict[str, bool]) -> Dict[str, str]:
    classes = {name: "backend-local" for name, _ in SEMANTIC_CLASSES}

    if kind == "c":
        if traits["emitc_pipeline"]:
            for name, _ in SEMANTIC_CLASSES[:8]:
                classes[name] = "shared"
        classes["malformed_input_diagnostic_text"] = "backend-local"
        if traits["emitc_pipeline"] and traits["schema_scan"] and traits["schema_selection_guard"]:
            classes["lowered_contract_validation"] = "shared"
        return classes

    if kind == "native":
        if traits["shared_pipeline"]:
            for name, _ in SEMANTIC_CLASSES[:8]:
                classes[name] = "shared"
        if traits["helper_binding_render"]:
            classes["malformed_input_diagnostic_text"] = "shared"
        if traits["collect_lowered_facts"]:
            classes["lowered_contract_validation"] = "shared"
        return classes

    if kind == "scripted":
        if traits["runtime_plan"]:
            classes["field_step_ordering"] = "shared"
        if traits["union_helper_usage"] and traits["helper_binding_render"]:
            classes["union_tag_mask_validate"] = "shared"
        if traits["scalar_helper_usage"] and traits["helper_binding_render"]:
            classes["scalar_normalize_sign_extend"] = "shared"
        if traits["array_helper_usage"]:
            classes["variable_array_prefix_validate"] = "shared"
            classes["fixed_array_cardinality_validate"] = "shared"
        if traits["delimiter_helper_usage"]:
            classes["delimited_payload_validate"] = "shared"
        if traits["capacity_helper_usage"]:
            classes["section_capacity_precheck"] = "shared"
        if traits["scripted_body_plan"]:
            classes["alignment_padding_orchestration"] = "shared"
        if traits["diagnostic_catalog"]:
            classes["malformed_input_diagnostic_text"] = "shared"
        if traits["collect_lowered_facts"]:
            classes["lowered_contract_validation"] = "shared"
        return classes

    raise ValueError(f"unknown backend kind: {kind}")


def _build_report(repo_root: Path) -> Dict[str, object]:
    backends: Dict[str, object] = {}
    per_backend_scores: Dict[str, int] = {}
    total_classes = len(SEMANTIC_CLASSES)

    for backend_name, cfg in BACKEND_CONFIG.items():
        emitter_text = _load_text(repo_root, cfg["path"])
        traits = _detect_traits(cfg["kind"], emitter_text)
        classifications = _classifications(cfg["kind"], traits)
        shared_count = sum(1 for status in classifications.values() if status == "shared")
        score = int(round((shared_count * 100.0) / total_classes))
        per_backend_scores[backend_name] = score

        backends[backend_name] = {
            "kind": cfg["kind"],
            "emitter": cfg["path"],
            "score": score,
            "shared_classes": shared_count,
            "total_classes": total_classes,
            "classifications": classifications,
            "traits": traits,
        }

    return {
        "version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "semantic_classes": [{"id": class_id, "label": label} for class_id, label in SEMANTIC_CLASSES],
        "backends": backends,
        "project_floor_score": min(per_backend_scores.values()),
    }


def _write_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_markdown(path: Path, report: Dict[str, object]) -> None:
    backends = report["backends"]
    lines: List[str] = []
    lines.append("# Convergence Scorecard")
    lines.append("")
    lines.append("Generated by `tools/convergence/convergence_report.py`.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Backend | Score | Shared / Total |")
    lines.append("| --- | ---: | ---: |")
    for backend_name in sorted(backends):
        backend = backends[backend_name]
        lines.append(
            f"| `{backend_name}` | `{backend['score']}` | `{backend['shared_classes']} / {backend['total_classes']}` |"
        )
    lines.append("")
    lines.append(f"Project floor score: `{report['project_floor_score']}`")
    lines.append("")
    lines.append("## Per-Backend Classifications")
    lines.append("")
    class_labels = {entry["id"]: entry["label"] for entry in report["semantic_classes"]}
    for backend_name in sorted(backends):
        backend = backends[backend_name]
        lines.append(f"### `{backend_name}`")
        lines.append("")
        lines.append("| Semantic class | Status |")
        lines.append("| --- | --- |")
        for class_id, _ in SEMANTIC_CLASSES:
            lines.append(f"| `{class_labels[class_id]}` | `{backend['classifications'][class_id]}` |")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _check_baseline(report: Dict[str, object], baseline_path: Path) -> List[str]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    failures: List[str] = []
    baseline_expected = baseline.get("expected", {})
    report_backends = report["backends"]

    for backend_name, expected_classes in baseline_expected.items():
        if backend_name not in report_backends:
            failures.append(f"baseline backend missing from report: {backend_name}")
            continue
        actual_classes = report_backends[backend_name]["classifications"]
        for class_id, expected_status in expected_classes.items():
            actual_status = actual_classes.get(class_id)
            if expected_status == "shared" and actual_status != "shared":
                failures.append(
                    f"regression: backend '{backend_name}' class '{class_id}' expected shared but found {actual_status}"
                )

    minimum_scores = baseline.get("minimum_scores", {})
    for backend_name, minimum_score in minimum_scores.get("backends", {}).items():
        if backend_name not in report_backends:
            failures.append(f"minimum score references unknown backend: {backend_name}")
            continue
        actual_score = report_backends[backend_name]["score"]
        if actual_score < int(minimum_score):
            failures.append(
                f"score regression: backend '{backend_name}' score {actual_score} is below minimum {minimum_score}"
            )

    project_floor_min = minimum_scores.get("project_floor")
    if project_floor_min is not None:
        actual_project_floor = int(report["project_floor_score"])
        if actual_project_floor < int(project_floor_min):
            failures.append(
                f"score regression: project floor {actual_project_floor} is below minimum {project_floor_min}"
            )
    return failures


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate convergence score report for llvm-dsdl backends.")
    parser.add_argument("--repo-root", required=True, help="Path to repository root.")
    parser.add_argument("--output-json", help="Path for JSON report output.")
    parser.add_argument("--output-md", help="Path for Markdown scorecard output.")
    parser.add_argument("--baseline", help="Path to baseline JSON used for regression checks.")
    parser.add_argument(
        "--check-regressions",
        action="store_true",
        help="Fail if baseline shared classifications regress or score minimums are violated.",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    report = _build_report(repo_root)

    if args.output_json:
        _write_json(Path(args.output_json), report)
    if args.output_md:
        _write_markdown(Path(args.output_md), report)

    if args.check_regressions:
        if not args.baseline:
            print("error: --check-regressions requires --baseline", file=sys.stderr)
            return 2
        baseline_path = Path(args.baseline).resolve()
        if not baseline_path.exists():
            print(f"error: baseline not found: {baseline_path}", file=sys.stderr)
            return 2
        failures = _check_baseline(report, baseline_path)
        if failures:
            for failure in failures:
                print(f"convergence regression: {failure}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
