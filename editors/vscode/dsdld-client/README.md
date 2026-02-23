# dsdld VS Code Client

This extension provides:

1. DSDL language registration for `.dsdl` files.
2. `.dsdl` file icon sourced from the repository root `dsdl.svg`.
3. LSP client bootstrap to `dsdld` over stdio.
4. Semantic-token highlighting from the language server.
5. Lifecycle commands for language-server restart/shutdown.

## Prerequisites

1. VS Code `>= 1.85`.
2. Built `dsdld` binary.
3. Node/npm for extension install and test runs.

## Install And Run

From this folder:

```bash
npm install
```

Set `dsdld.serverPath` in VS Code settings, or set environment variable `DSDLD_BINARY`.
If neither is set, the extension tries `dsdld` from `PATH`.

## Settings

The extension contributes these settings:

1. `dsdld.serverPath`: absolute path to `dsdld` (optional).
2. `dsdld.rootNamespaceDirs`: root namespace directories.
3. `dsdld.lookupDirs`: dependency lookup directories.
4. `dsdld.lintEnabled`: enable/disable lint diagnostics.
5. `dsdld.aiMode`: `off`, `suggest`, `assist`, `apply_with_confirmation`.
6. `dsdld.trace`: `off`, `basic`, `verbose`.

Example workspace settings:

```json
{
  "dsdld.serverPath": "/absolute/path/to/dsdld",
  "dsdld.rootNamespaceDirs": [
    "/absolute/path/to/root_namespace"
  ],
  "dsdld.lookupDirs": [
    "/absolute/path/to/lookup_namespace"
  ],
  "dsdld.lintEnabled": true,
  "dsdld.aiMode": "off",
  "dsdld.trace": "basic"
}
```

## Commands

Command palette entries:

1. `DSDLD: Restart Language Server`
2. `DSDLD: Shutdown Language Server`

## Debug Flow

1. Open this folder (`editors/vscode/dsdld-client`) in VS Code.
2. Ensure `DSDLD_BINARY` points to your built `dsdld` binary.
3. Press `F5` and run `Run Extension`.
4. Open a `.dsdl` file in the Extension Development Host window.
5. Check diagnostics, hover, definition, completion, semantic tokens, rename.

The debug launch configs are in:

- `.vscode/launch.json`

## Test

Run smoke tests from this folder:

```bash
DSDLD_BINARY=/absolute/path/to/dsdld npm test
```

The test runner already sets short temp directories for VS Code user data and extensions to avoid long IPC path issues on macOS.
