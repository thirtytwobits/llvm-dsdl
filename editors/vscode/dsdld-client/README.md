# dsdld VSCode Client

This extension provides:

1. DSDL language registration for `.dsdl` files.
2. LSP client bootstrap to `dsdld` over stdio.
3. Semantic-token-based highlighting from the language server.

## Settings

The extension contributes:

1. `dsdld.serverPath`: absolute path to `dsdld` binary (optional).
2. `dsdld.rootNamespaceDirs`: root namespace directories.
3. `dsdld.lookupDirs`: lookup directories.
4. `dsdld.lintEnabled`: enable linting.
5. `dsdld.aiEnabled`: enable AI features.
6. `dsdld.trace`: trace verbosity (`off`, `basic`, `verbose`).

If `dsdld.serverPath` is empty, the extension tries `DSDLD_BINARY`, then `dsdld` from `PATH`.

## Test

Run from this folder:

```bash
npm install
DSDLD_BINARY=/absolute/path/to/dsdld npm test
```

## Commands

The extension contributes these command palette actions:

1. `DSDLD: Restart Language Server`
2. `DSDLD: Shutdown Language Server`
