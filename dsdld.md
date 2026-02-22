# dsdld Execution Backlog

## 1. Objective

Build a production Language Server Protocol implementation at:

- `tools/dsdld`

with a first-party VSCode client extension, using `llvm-dsdl` core APIs and targeting clangd-class workflows:

1. Whole-namespace indexing
2. Sophisticated completion/symbol ranking and reranking
3. Refactoring and fix-its
4. Extensible linting engine
5. Optional agentic AI backend integration with strict safety controls

## 2. Product Outcomes

At completion:

1. `dsdld` provides fast, incremental diagnostics on edit.
2. VSCode has first-party DSDL language support with syntax coloration.
3. Semantic navigation works across DSDL namespaces and versions.
4. Workspace-scale symbol and reference indexing is stable.
5. Refactors are safe, previewable, and conflict-aware.
6. Lint rules are configurable and can provide autofixes.
7. AI assist is optional and never required for core deterministic behavior.

## 3. Core Constraints

1. The server must remain fully functional with AI disabled.
2. All write operations must be explicit LSP workspace edits.
3. Indexing and analysis must be cancellation-aware.
4. Linux/macOS first; keep abstractions clean for Windows follow-up.
5. Prefer reuse of existing compiler APIs:
   - `parseDefinitions(...)`
   - `analyze(...)`
   - `DiagnosticEngine`
   - optional `lowerToMLIR(...)` for deep semantic tools

## 4. High-Level Architecture

1. `tools/dsdld`:
   - JSON-RPC/LSP transport over stdio.
2. VSCode extension client (`editors/vscode/dsdld-client`):
   - language registration for `.dsdl`
   - semantic token bridge
   - LSP client/session bootstrap
3. `lib/LSP` subsystem:
   - protocol layer, session manager, request scheduler.
4. `lib/LSP/Analysis`:
   - overlay-aware parse/analyze pipeline wrapper.
5. `lib/LSP/Index`:
   - shard builder, merged index, query engine.
6. `lib/LSP/Refactor`:
   - rename/workspace edit planner and conflict detector.
7. `lib/LSP/Lint`:
   - rule registry, execution engine, fix-it provider.
8. `lib/LSP/AI`:
   - provider abstraction, policy gates, prompt context packer.

## 5. Milestones

1. M0: `dsdld` scaffolding and protocol skeleton
2. M1: VSCode extension bootstrap and DSDL coloration
3. M2: diagnostics + semantic navigation baseline
4. M3: whole-namespace index + symbol/references performance
5. M4: completion heuristics and reranking
6. M5: refactoring and fix-its
7. M6: lint engine and initial ruleset
8. M7: agentic AI integration (opt-in)
9. M8: hardening, benchmarks, docs, release readiness

## 6. Epic Backlog

### Epic A: Server Foundation

Status: Completed on February 21, 2026.

| ID | Task | Priority | Depends On | Exit Criteria |
|---|---|---|---|---|
| A1 | Create `tools/dsdld` target and stdio JSON-RPC loop | P0 | none | Server responds to `initialize`, `shutdown`, `exit` |
| A2 | Add session state (`DocumentStore`, open-file overlay map) | P0 | A1 | Open/change/close roundtrip tests pass |
| A3 | Add request scheduler with cancellation tokens | P0 | A1 | Long-running requests cancel cleanly |
| A4 | Add config ingestion (roots/lookups, lint toggles, ai mode) | P1 | A2 | `workspace/didChangeConfiguration` supported |
| A5 | Add tracing and metrics hooks | P1 | A1 | Request latency telemetry emitted |

### Epic B: VSCode Plugin and DSDL Coloration

Status: Completed on February 21, 2026.

| ID | Task | Priority | Depends On | Exit Criteria |
|---|---|---|---|---|
| B1 | Create `editors/vscode/dsdld-client` extension scaffold (`package.json`, activation, client bootstrap) | P0 | A1 | Extension activates and starts `dsdld` |
| B2 | Add DSDL language registration and file associations (`.dsdl`) | P0 | B1 | Files open in DSDL language mode |
| B3 | Wire semantic tokens from LSP highlighting | P0 | B2, A1 | Keywords/types/directives/tokens are colored in VSCode |
| B4 | Stabilize semantic-token-only highlighting behavior | P1 | B3 | Highlighting remains accurate without TextMate grammar fallback |
| B5 | Add extension settings for root/lookup dirs and server path | P1 | B1, A4 | Settings update LSP startup behavior |
| B6 | Add VSCode integration tests (`vscode-test`) for startup, diagnostics, and coloring smoke | P1 | B3, B4 | CI integration test lane passes |

### Epic C: Analysis Pipeline Integration

Status: Completed on February 21, 2026.

| ID | Task | Priority | Depends On | Exit Criteria |
|---|---|---|---|---|
| C1 | Create overlay-aware compilation adapter around parse/analyze | P0 | A2 | Unsaved edits reflected in diagnostics |
| C2 | Add incremental invalidation graph (file -> impacted defs) | P0 | C1 | Dirty-file updates avoid full rebuild when possible |
| C3 | Map `DiagnosticEngine` output to LSP diagnostics | P0 | C1 | Accurate range/severity mapping in editor |
| C4 | Add semantic snapshot cache with versioning | P1 | C1 | Stale snapshot use is prevented by version checks |
| C5 | Add optional MLIR snapshot path for advanced tools | P2 | C4 | MLIR data available behind feature flag |

### Epic D: Baseline Language Features

Status: Completed on February 21, 2026.

| ID | Task | Priority | Depends On | Exit Criteria |
|---|---|---|---|---|
| D1 | `textDocument/hover` | P0 | C4 | Hover returns resolved type details and versions |
| D2 | `textDocument/definition` | P0 | C4 | Composite refs jump to correct source |
| D3 | `textDocument/references` | P0 | C4 | Cross-file references are stable |
| D4 | `textDocument/documentSymbol` | P1 | C4 | Symbols listed by kind with stable ranges |
| D5 | semantic tokens | P1 | D4, B4 | Token classes for type/field/const/directive |
| D6 | baseline completion | P1 | D1 | Completion offers valid symbols in context |

### Epic E: Whole-Namespace Indexing

Status: Completed on February 21, 2026.

| ID | Task | Priority | Depends On | Exit Criteria |
|---|---|---|---|---|
| E1 | Define index schema (symbols, refs, metadata) | P0 | C4 | Versioned schema committed and documented |
| E2 | Build per-file shard writer/reader | P0 | E1 | Shards generated deterministically |
| E3 | Build merged workspace index with warm cache load | P0 | E2 | Warm startup index load under target |
| E4 | Background indexing with throttling and cancellation | P1 | E2 | Reindex does not block interactive requests |
| E5 | Workspace symbol query API | P1 | E3 | `workspace/symbol` latency target met |
| E6 | Index consistency verifier and repair path | P2 | E3 | Corrupt shard handling tested |

### Epic F: Heuristics and Re-ranking

Status: Completed on February 21, 2026.

| ID | Task | Priority | Depends On | Exit Criteria |
|---|---|---|---|---|
| F1 | Define ranking features and score model | P0 | D6, E5 | Feature spec with explainable scoring |
| F2 | Implement completion reranker | P0 | F1 | Relevance benchmark improves over baseline |
| F3 | Implement symbol search reranker | P1 | F1 | Workspace symbol precision improves |
| F4 | Add recency/frequency adaptive signals | P2 | F2 | Signals persisted and bounded |
| F5 | Add `score-explain` debug endpoint | P2 | F2 | Ranking decisions introspectable |

### Epic G: Refactoring and Fix-Its

Status: Completed on February 21, 2026.

| ID | Task | Priority | Depends On | Exit Criteria |
|---|---|---|---|---|
| G1 | Implement `prepareRename` + `rename` | P0 | D2, D3, E3 | Multi-file rename with conflict detection |
| G2 | Workspace edit planner with dry-run preview | P0 | G1 | Preview edits match applied edits |
| G3 | Diagnostic fix-it plumbing from core diagnostics | P1 | C3 | LSP code actions attach precise edits |
| G4 | Initial fix-it set (unresolved composite, service field misuse, root/lookup misconfig) | P1 | G3 | Fix-its validated by integration tests |
| G5 | Refactor code actions (`extract type`, `normalize version refs`) | P2 | G2 | Refactor tests pass on corpus fixtures |

### Epic H: Lint Engine (Extensible)

Status: Completed on February 21, 2026.

| ID | Task | Priority | Depends On | Exit Criteria |
|---|---|---|---|---|
| H1 | Create rule API and execution engine | P0 | C4 | Rules register/execute deterministically |
| H2 | Config schema and suppression model | P0 | H1 | Per-workspace and per-file suppression works |
| H3 | Baseline ruleset v1 | P1 | H1 | At least 10 useful rules with docs |
| H4 | Autofix-capable lint rules | P1 | H1 | Autofix emits valid workspace edits |
| H5 | Lint rule test harness + golden diagnostics | P1 | H1 | Rule regressions caught in CI |
| H6 | Rule plug-in extension path | P2 | H1 | External/internal rule packs loadable |

### Epic I: Agentic AI Integration

| ID | Task | Priority | Depends On | Exit Criteria |
|---|---|---|---|---|
| I1 | AI provider abstraction (`OFF`, `SUGGEST`, `ASSIST`, `APPLY_WITH_CONFIRMATION`) | P0 | A4 | Mode switching verified |
| I2 | Context packer (symbol/index/diagnostic aware) | P1 | E3, C4 | Prompts use bounded, relevant context |
| I3 | Safe edit policy gate + confirmation flow | P0 | G2, I1 | AI never writes without explicit confirmation |
| I4 | AI-assisted code actions (explain/fix/suggest) | P1 | I2, I3 | Suggestions degrade gracefully when offline |
| I5 | Audit logging/redaction and policy tests | P1 | I1 | No secret leakage in logs; tests enforce |
| I6 | Tool-use interface for agentic workflows | P2 | I2 | Agent tasks can call safe local actions |

### Epic J: Performance, Reliability, and Scale

| ID | Task | Priority | Depends On | Exit Criteria |
|---|---|---|---|---|
| J1 | LSP replay harness for latency regression | P0 | A1 | p50/p95/p99 tracked in CI artifact |
| J2 | Large-corpus stress suite (`test/benchmark/complex`) | P0 | E4 | No crashes, bounded memory growth |
| J3 | Cancellation storm and file-churn robustness tests | P1 | A3 | No deadlocks or stale publish loops |
| J4 | Index cold/warm benchmarks | P1 | E3 | Targets met and documented |
| J5 | Fuzz and malformed JSON-RPC handling | P1 | A1 | Invalid input never crashes server |

### Epic K: Documentation and Adoption

| ID | Task | Priority | Depends On | Exit Criteria |
|---|---|---|---|---|
| K1 | `tools/dsdld/README.md` with VS Code/Neovim setup | P0 | M2 | Users can run server locally in minutes |
| K2 | `editors/vscode/dsdld-client/README.md` and extension settings docs | P0 | M1 | Extension install + debug flow documented |
| K3 | `DESIGN.md` LSP architecture section | P1 | M3 | Architecture diagrams and data flow updated |
| K4 | Lint rule authoring guide | P1 | H1 | New rule can be added via documented steps |
| K5 | AI safety and operator guide | P1 | I1 | Modes and controls clearly documented |
| K6 | Release checklist updates for dsdld + VSCode extension | P1 | M8 | QA gates include LSP and extension checks |

## 7. Delivery Sequence (Execution Order)

1. Sprint 1: A + B1/B2/B3 + C1/C3
2. Sprint 2: B4/B5/B6 + C2/C4 + D1/D2/D3
3. Sprint 3: D4/D5/D6 + E1/E2/E3
4. Sprint 4: E4/E5 + F1/F2 + G1/G2
5. Sprint 5: G3/G4 + H1/H2/H3
6. Sprint 6: H4/H5 + I1/I2/I3
7. Sprint 7: I4/I5 + J1/J2/J3
8. Sprint 8: J4/J5 + K1/K2/K3/K4/K5/K6 + hardening

## 8. Definition of Done

1. `tools/dsdld` passes unit + integration + replay performance gates.
2. VSCode extension supports:
   - `.dsdl` language mode
   - syntax coloration
   - semantic highlighting
   - diagnostics + hover + go-to-definition
3. Full workspace indexing and symbol queries are stable on the complex benchmark corpus.
4. Rename and fix-it workflows are validated on multi-file edits.
5. Lint engine supports configuration, suppression, and autofix.
6. AI mode is optional, policy-gated, and covered by safety tests.
7. Docs are complete for both server and extension adoption.

## 9. Initial Implementation Tasks (Start Immediately)

1. Create CMake target and folder skeleton under `tools/dsdld`.
2. Add JSON-RPC stdio loop with `initialize/shutdown/exit`.
3. Create `editors/vscode/dsdld-client` extension scaffold and DSDL language registration.
4. Add semantic-token highlighting for DSDL coloration.
5. Wire parse/analyze pipeline and map `DiagnosticEngine` to LSP diagnostics.
6. Add `definition` and `hover` using semantic model.

## 10. Risk Register

1. Overlay parsing gap in current API:
   - Mitigation: add overlay-aware parse entrypoint early (Epic C).
2. Index memory growth on large corpora:
   - Mitigation: shard format + mmap + compaction strategy (Epic E, J).
3. Refactor correctness across versioned types:
   - Mitigation: stable symbol IDs + rename dry-run diff validator (Epic G).
4. VSCode highlight drift from parser/AST semantics:
   - Mitigation: golden token tests for semantic tokens (Epic B, D).
5. AI safety drift:
   - Mitigation: hard policy gate and audit tests (Epic I).
