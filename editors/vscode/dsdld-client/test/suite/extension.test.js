//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

const assert = require("assert");
const fs = require("fs");
const os = require("os");
const path = require("path");
const vscode = require("vscode");

function sleep(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function waitForDiagnostics(uri, minimumCount) {
  for (let attempt = 0; attempt < 120; attempt += 1) {
    const diagnostics = vscode.languages.getDiagnostics(uri);
    if (diagnostics.length >= minimumCount) {
      return diagnostics;
    }
    await sleep(50);
  }
  return vscode.languages.getDiagnostics(uri);
}

suite("dsdld-client", () => {
  test("activates and registers .dsdl language mode", async () => {
    const extension = vscode.extensions.getExtension("opencyphal.dsdld-client");
    assert.ok(extension, "extension not found");
    await extension.activate();

    const tempFile = path.join(
      os.tmpdir(),
      `dsdld-client-language-${Date.now()}.dsdl`
    );
    fs.writeFileSync(tempFile, "uint8 value\n@sealed\n", "utf8");

    const doc = await vscode.workspace.openTextDocument(
      vscode.Uri.file(tempFile)
    );
    assert.strictEqual(doc.languageId, "dsdl");
  });

  test("publishes diagnostics for invalid field type", async () => {
    const doc = await vscode.workspace.openTextDocument({
      language: "dsdl",
      content: "definitely_invalid_type field_name\n@sealed\n",
    });
    await vscode.window.showTextDocument(doc);

    const diagnostics = await waitForDiagnostics(doc.uri, 1);
    assert.ok(diagnostics.length > 0, "expected diagnostics");
    assert.ok(
      diagnostics[0].source === "dsdld",
      "expected diagnostics from dsdld"
    );
    assert.ok(
      diagnostics.some((diagnostic) => diagnostic.severity === vscode.DiagnosticSeverity.Error),
      "expected diagnostics from dsdld"
    );
  });

  test("returns semantic tokens for dsdl document", async () => {
    const doc = await vscode.workspace.openTextDocument({
      language: "dsdl",
      content:
        "@union\nuint8 first\nuint16 second\n---\n# comment\n@sealed\n",
    });
    await vscode.window.showTextDocument(doc);

    const semanticTokens = await vscode.commands.executeCommand(
      "vscode.provideDocumentSemanticTokens",
      doc.uri
    );
    assert.ok(semanticTokens, "semantic token result missing");
    assert.ok(
      semanticTokens.data && semanticTokens.data.length > 0,
      "expected non-empty semantic token payload"
    );
  });

  test("registers lifecycle commands", async () => {
    const commands = await vscode.commands.getCommands(true);
    assert.ok(
      commands.includes("dsdld.restartLanguageServer"),
      "missing dsdld.restartLanguageServer command"
    );
    assert.ok(
      commands.includes("dsdld.shutdownLanguageServer"),
      "missing dsdld.shutdownLanguageServer command"
    );
  });
});
