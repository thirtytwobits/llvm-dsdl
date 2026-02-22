#!/usr/bin/env python3
#===----------------------------------------------------------------------===#
#
# Part of the OpenCyphal project, under the MIT licence
# SPDX-License-Identifier: MIT
#
#===----------------------------------------------------------------------===#

"""LSP performance benchmark harness for replay and index cold/warm runs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    fraction = rank - low
    return sorted_values[low] * (1.0 - fraction) + sorted_values[high] * fraction


def summarize_latencies(samples_ms: list[float]) -> dict[str, float]:
    return {
        "count": float(len(samples_ms)),
        "min_ms": min(samples_ms) if samples_ms else 0.0,
        "max_ms": max(samples_ms) if samples_ms else 0.0,
        "p50_ms": percentile(samples_ms, 0.50),
        "p95_ms": percentile(samples_ms, 0.95),
        "p99_ms": percentile(samples_ms, 0.99),
    }


def path_to_uri(path: Path) -> str:
    return path.resolve().as_uri()


@dataclass(frozen=True)
class DocumentFixture:
    """One opened document fixture in the replay/index sessions."""

    path: Path
    uri: str
    text: str


class LspProcess:
    """Minimal stdio LSP client for benchmarking request latencies."""

    def __init__(self, executable: Path, request_timeout_sec: float) -> None:
        self._executable = executable
        self._request_timeout_sec = request_timeout_sec
        self._proc: subprocess.Popen[bytes] | None = None
        self._reader: threading.Thread | None = None
        self._lock = threading.Condition()
        self._responses: dict[int, dict[str, Any]] = {}
        self._notifications: list[dict[str, Any]] = []
        self._next_id = 1
        self._reader_error: str | None = None
        self._stopped = False

    def start(self) -> None:
        self._proc = subprocess.Popen(
            [str(self._executable)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("failed to open dsdld stdio pipes")
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

    def _read_message(self) -> dict[str, Any] | None:
        assert self._proc is not None
        assert self._proc.stdout is not None
        stream = self._proc.stdout

        headers: dict[str, str] = {}
        saw_headers = False
        while True:
            line = stream.readline()
            if not line:
                return None
            line = line.rstrip(b"\r\n")
            if not line:
                break
            saw_headers = True
            if b":" not in line:
                continue
            key, value = line.split(b":", 1)
            headers[key.decode("ascii", errors="ignore").strip().lower()] = value.decode(
                "ascii", errors="ignore"
            ).strip()

        if not saw_headers:
            return None
        if "content-length" not in headers:
            raise RuntimeError("missing Content-Length header")
        content_length = int(headers["content-length"])
        payload = stream.read(content_length)
        if len(payload) != content_length:
            raise RuntimeError("truncated LSP payload")
        return json.loads(payload.decode("utf-8"))

    def _reader_loop(self) -> None:
        try:
            while True:
                message = self._read_message()
                if message is None:
                    break
                with self._lock:
                    if "id" in message and ("result" in message or "error" in message):
                        message_id = int(message["id"])
                        self._responses[message_id] = message
                    else:
                        self._notifications.append(message)
                    self._lock.notify_all()
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._reader_error = str(exc)
                self._lock.notify_all()
        finally:
            with self._lock:
                self._stopped = True
                self._lock.notify_all()

    def _send(self, message: dict[str, Any]) -> None:
        assert self._proc is not None
        assert self._proc.stdin is not None
        payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
        frame = b"Content-Length: " + str(len(payload)).encode("ascii") + b"\r\n\r\n" + payload
        self._proc.stdin.write(frame)
        self._proc.stdin.flush()

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self._send(message)

    def request(self, method: str, params: dict[str, Any] | None = None) -> tuple[dict[str, Any], float]:
        with self._lock:
            request_id = self._next_id
            self._next_id += 1

        message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params

        started = time.perf_counter()
        self._send(message)
        deadline = started + self._request_timeout_sec
        with self._lock:
            while request_id not in self._responses and not self._stopped and self._reader_error is None:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError(f"timed out waiting for response to {method}")
                self._lock.wait(timeout=remaining)
            if self._reader_error:
                raise RuntimeError(f"LSP reader failed: {self._reader_error}")
            if request_id not in self._responses:
                raise RuntimeError(f"LSP process stopped before responding to {method}")
            response = self._responses.pop(request_id)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return response, elapsed_ms

    def close(self) -> None:
        if not self._proc:
            return
        try:
            if self._proc.poll() is None:
                self._proc.terminate()
                self._proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        finally:
            stderr_text = ""
            if self._proc.stderr is not None:
                stderr_text = self._proc.stderr.read().decode("utf-8", errors="replace")
                if stderr_text.strip():
                    print("[lsp-bench][dsdld stderr]")
                    print(stderr_text.rstrip())


def collect_documents(root_namespace_dir: Path, open_file_count: int) -> list[DocumentFixture]:
    documents: list[DocumentFixture] = []
    for path in sorted(root_namespace_dir.rglob("*.dsdl"))[:open_file_count]:
        text = path.read_text(encoding="utf-8")
        documents.append(DocumentFixture(path=path, uri=path_to_uri(path), text=text))
    if not documents:
        raise FileNotFoundError(f"no .dsdl documents found under {root_namespace_dir}")
    return documents


def apply_workspace_settings(
    lsp: LspProcess,
    root_namespace_dir: Path,
    lookup_dirs: list[Path],
    index_cache_dir: Path,
) -> None:
    lsp.notify(
        "workspace/didChangeConfiguration",
        {
            "settings": {
                "roots": [str(root_namespace_dir)],
                "lookupDirs": [str(path) for path in lookup_dirs],
                "indexCacheDir": str(index_cache_dir),
                "trace": "off",
            }
        },
    )


def initialize_session(
    lsp: LspProcess,
    root_namespace_dir: Path,
    lookup_dirs: list[Path],
    index_cache_dir: Path,
    documents: list[DocumentFixture],
) -> None:
    response, _ = lsp.request("initialize", {})
    if "error" in response:
        raise RuntimeError(f"initialize failed: {response['error']}")
    apply_workspace_settings(lsp, root_namespace_dir, lookup_dirs, index_cache_dir)
    for index, document in enumerate(documents):
        lsp.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": document.uri,
                    "languageId": "dsdl",
                    "version": index + 1,
                    "text": document.text,
                }
            },
        )


def shutdown_session(lsp: LspProcess) -> None:
    try:
        lsp.request("shutdown", None)
    except Exception:  # noqa: BLE001
        pass
    try:
        lsp.notify("exit", None)
    finally:
        lsp.close()


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    dsdld = Path(args.dsdld).resolve()
    root_namespace_dir = Path(args.root_namespace_dir).resolve()
    lookup_dirs = [Path(path).resolve() for path in args.lookup_dir]
    index_cache_dir = Path(args.index_cache_dir).resolve()
    index_cache_dir.mkdir(parents=True, exist_ok=True)
    documents = collect_documents(root_namespace_dir, args.open_file_count)

    lsp = LspProcess(dsdld, request_timeout_sec=args.request_timeout_sec)
    lsp.start()
    per_method_samples: dict[str, list[float]] = {
        "textDocument/documentSymbol": [],
        "textDocument/completion": [],
        "textDocument/hover": [],
        "textDocument/definition": [],
        "workspace/symbol": [],
    }
    errors: list[str] = []

    try:
        initialize_session(lsp, root_namespace_dir, lookup_dirs, index_cache_dir, documents)
        for iteration in range(args.request_iterations):
            document = documents[iteration % len(documents)]
            requests: list[tuple[str, dict[str, Any]]] = [
                ("textDocument/documentSymbol", {"textDocument": {"uri": document.uri}}),
                (
                    "textDocument/completion",
                    {"textDocument": {"uri": document.uri}, "position": {"line": 0, "character": 1}},
                ),
                ("textDocument/hover", {"textDocument": {"uri": document.uri}, "position": {"line": 0, "character": 1}}),
                (
                    "textDocument/definition",
                    {"textDocument": {"uri": document.uri}, "position": {"line": 0, "character": 1}},
                ),
                ("workspace/symbol", {"query": args.workspace_symbol_query}),
            ]
            for method, params in requests:
                response, elapsed_ms = lsp.request(method, params)
                if "error" in response:
                    errors.append(f"{method}: {response['error']}")
                    continue
                per_method_samples[method].append(elapsed_ms)
    finally:
        shutdown_session(lsp)

    aggregate_samples = [sample for samples in per_method_samples.values() for sample in samples]
    metrics = {
        method: summarize_latencies(samples) for method, samples in per_method_samples.items()
    }
    report = {
        "schema_version": 1,
        "suite_name": "llvmdsdl-lsp-replay",
        "created_utc": utc_now(),
        "config": {
            "dsdld": str(dsdld),
            "root_namespace_dir": str(root_namespace_dir),
            "lookup_dirs": [str(path) for path in lookup_dirs],
            "index_cache_dir": str(index_cache_dir),
            "open_file_count": args.open_file_count,
            "request_iterations": args.request_iterations,
            "workspace_symbol_query": args.workspace_symbol_query,
        },
        "aggregate_latency_ms": summarize_latencies(aggregate_samples),
        "methods": metrics,
        "error_count": len(errors),
        "errors": errors[:32],
    }
    return report


def run_index_phase(
    dsdld: Path,
    root_namespace_dir: Path,
    lookup_dirs: list[Path],
    index_cache_dir: Path,
    documents: list[DocumentFixture],
    request_timeout_sec: float,
    symbol_query: str,
    symbol_iterations: int,
) -> dict[str, Any]:
    lsp = LspProcess(dsdld, request_timeout_sec=request_timeout_sec)
    lsp.start()
    samples: list[float] = []
    index_stats: dict[str, Any] = {}
    errors: list[str] = []

    try:
        initialize_session(lsp, root_namespace_dir, lookup_dirs, index_cache_dir, documents)
        for _ in range(symbol_iterations):
            response, elapsed_ms = lsp.request("workspace/symbol", {"query": symbol_query})
            if "error" in response:
                errors.append(str(response["error"]))
                continue
            samples.append(elapsed_ms)

        response, _ = lsp.request("dsdld/debug/indexStats", {})
        if "result" in response and isinstance(response["result"], dict):
            index_stats = response["result"]
        elif "error" in response:
            errors.append(str(response["error"]))
    finally:
        shutdown_session(lsp)

    return {
        "latency_ms": summarize_latencies(samples),
        "index_stats": index_stats,
        "error_count": len(errors),
        "errors": errors[:16],
    }


def run_index_bench(args: argparse.Namespace) -> dict[str, Any]:
    dsdld = Path(args.dsdld).resolve()
    root_namespace_dir = Path(args.root_namespace_dir).resolve()
    lookup_dirs = [Path(path).resolve() for path in args.lookup_dir]
    index_cache_dir = Path(args.index_cache_dir).resolve()
    documents = collect_documents(root_namespace_dir, args.open_file_count)

    shutil.rmtree(index_cache_dir, ignore_errors=True)
    index_cache_dir.mkdir(parents=True, exist_ok=True)

    cold = run_index_phase(
        dsdld=dsdld,
        root_namespace_dir=root_namespace_dir,
        lookup_dirs=lookup_dirs,
        index_cache_dir=index_cache_dir,
        documents=documents,
        request_timeout_sec=args.request_timeout_sec,
        symbol_query=args.workspace_symbol_query,
        symbol_iterations=args.symbol_iterations,
    )
    warm = run_index_phase(
        dsdld=dsdld,
        root_namespace_dir=root_namespace_dir,
        lookup_dirs=lookup_dirs,
        index_cache_dir=index_cache_dir,
        documents=documents,
        request_timeout_sec=args.request_timeout_sec,
        symbol_query=args.workspace_symbol_query,
        symbol_iterations=args.symbol_iterations,
    )

    report = {
        "schema_version": 1,
        "suite_name": "llvmdsdl-lsp-index-cold-warm",
        "created_utc": utc_now(),
        "config": {
            "dsdld": str(dsdld),
            "root_namespace_dir": str(root_namespace_dir),
            "lookup_dirs": [str(path) for path in lookup_dirs],
            "index_cache_dir": str(index_cache_dir),
            "open_file_count": args.open_file_count,
            "symbol_iterations": args.symbol_iterations,
            "workspace_symbol_query": args.workspace_symbol_query,
        },
        "phases": {
            "cold": cold,
            "warm": warm,
        },
        "delta_ms": {
            "p50_ms": cold["latency_ms"]["p50_ms"] - warm["latency_ms"]["p50_ms"],
            "p95_ms": cold["latency_ms"]["p95_ms"] - warm["latency_ms"]["p95_ms"],
            "p99_ms": cold["latency_ms"]["p99_ms"] - warm["latency_ms"]["p99_ms"],
        },
    }
    return report


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_replay_summary(report: dict[str, Any]) -> None:
    print("[lsp-replay] summary")
    aggregate = report["aggregate_latency_ms"]
    print(
        f"  aggregate p50/p95/p99 (ms): "
        f"{aggregate['p50_ms']:.2f}/{aggregate['p95_ms']:.2f}/{aggregate['p99_ms']:.2f}"
    )
    print(f"  errors: {report['error_count']}")
    for method, metrics in report["methods"].items():
        print(
            f"  {method:<28} p50/p95/p99={metrics['p50_ms']:.2f}/{metrics['p95_ms']:.2f}/{metrics['p99_ms']:.2f} "
            f"count={int(metrics['count'])}"
        )


def print_index_summary(report: dict[str, Any]) -> None:
    print("[lsp-index] summary")
    cold = report["phases"]["cold"]["latency_ms"]
    warm = report["phases"]["warm"]["latency_ms"]
    print(
        f"  cold p50/p95/p99 (ms): {cold['p50_ms']:.2f}/{cold['p95_ms']:.2f}/{cold['p99_ms']:.2f}"
    )
    print(
        f"  warm p50/p95/p99 (ms): {warm['p50_ms']:.2f}/{warm['p95_ms']:.2f}/{warm['p99_ms']:.2f}"
    )
    print(
        f"  delta p50/p95/p99 (ms): "
        f"{report['delta_ms']['p50_ms']:.2f}/{report['delta_ms']['p95_ms']:.2f}/{report['delta_ms']['p99_ms']:.2f}"
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dsdld", required=True, help="Path to dsdld executable.")
    parser.add_argument("--root-namespace-dir", required=True, help="Root namespace directory for workspace analysis.")
    parser.add_argument(
        "--lookup-dir",
        action="append",
        default=[],
        help="Lookup directory (repeat for multiple).",
    )
    parser.add_argument("--index-cache-dir", required=True, help="Index cache directory path.")
    parser.add_argument("--report-json", required=True, help="Output report JSON path.")
    parser.add_argument(
        "--open-file-count",
        type=int,
        default=64,
        help="Maximum number of files to open into the benchmark session.",
    )
    parser.add_argument(
        "--request-timeout-sec",
        type=float,
        default=180.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--workspace-symbol-query",
        default="Vision",
        help="Query string used for workspace symbol benchmark requests.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark harness for dsdld replay/index performance")
    subparsers = parser.add_subparsers(dest="command", required=True)

    replay = subparsers.add_parser("replay", help="Run mixed-method LSP replay latency benchmark")
    add_common_args(replay)
    replay.add_argument(
        "--request-iterations",
        type=int,
        default=40,
        help="Replay iterations (each iteration issues a fixed mixed request set).",
    )

    index = subparsers.add_parser("index-bench", help="Run workspace symbol cold/warm index benchmark")
    add_common_args(index)
    index.add_argument(
        "--symbol-iterations",
        type=int,
        default=80,
        help="Number of workspace/symbol requests per phase.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "replay":
        report = run_replay(args)
        print_replay_summary(report)
        write_report(Path(args.report_json), report)
        return 0
    if args.command == "index-bench":
        report = run_index_bench(args)
        print_index_summary(report)
        write_report(Path(args.report_json), report)
        return 0
    raise RuntimeError(f"unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
