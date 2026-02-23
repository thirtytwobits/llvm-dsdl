//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

const vscode = require("vscode");
const { LanguageClient, TransportKind } = require("vscode-languageclient/node");

let client;
let configWatcher;
let lifecycleChain = Promise.resolve();
let serverOptions;
let clientOptions;

function getConfigValue(name, fallback) {
  return vscode.workspace.getConfiguration("dsdld").get(name, fallback);
}

function buildSettings() {
  const aiMode = getConfigValue("aiMode", "off");
  return {
    roots: getConfigValue("rootNamespaceDirs", []),
    lookupDirs: getConfigValue("lookupDirs", []),
    lint: { enabled: getConfigValue("lintEnabled", true) },
    ai: { mode: aiMode },
    trace: getConfigValue("trace", "basic"),
  };
}

function resolveServerCommand() {
  const configuredPath = getConfigValue("serverPath", "").trim();
  if (configuredPath.length > 0) {
    return configuredPath;
  }

  const envPath = (process.env.DSDLD_BINARY || "").trim();
  if (envPath.length > 0) {
    return envPath;
  }

  return "dsdld";
}

async function pushConfiguration() {
  if (!client) {
    return;
  }
  await client.sendNotification("workspace/didChangeConfiguration", {
    settings: buildSettings(),
  });
}

function runSerialLifecycleStep(step) {
  lifecycleChain = lifecycleChain.then(step, step);
  return lifecycleChain;
}

function assertClientOptionsInitialized() {
  if (!serverOptions || !clientOptions) {
    throw new Error("dsdld client options are not initialized");
  }
}

async function startLanguageServer() {
  assertClientOptionsInitialized();
  if (client) {
    return;
  }

  client = new LanguageClient(
    "dsdld",
    "DSDL Language Server",
    serverOptions,
    clientOptions
  );
  await client.start();
  await pushConfiguration();
}

async function stopLanguageServer() {
  if (!client) {
    return false;
  }
  const activeClient = client;
  client = undefined;
  await activeClient.stop();
  return true;
}

async function activate(context) {
  const serverCommand = resolveServerCommand();
  serverOptions = {
    command: serverCommand,
    transport: TransportKind.stdio,
  };

  clientOptions = {
    documentSelector: [
      { scheme: "file", language: "dsdl" },
      { scheme: "untitled", language: "dsdl" },
    ],
    initializationOptions: buildSettings(),
    synchronize: {
      configurationSection: "dsdld",
    },
    outputChannelName: "DSDLD",
  };

  await startLanguageServer();

  configWatcher = vscode.workspace.onDidChangeConfiguration(async (event) => {
    if (!event.affectsConfiguration("dsdld")) {
      return;
    }
    await pushConfiguration();
  });
  const restartCommand = vscode.commands.registerCommand(
    "dsdld.restartLanguageServer",
    async () => {
      await runSerialLifecycleStep(async () => {
        await stopLanguageServer();
        await startLanguageServer();
      });
      await vscode.window.showInformationMessage(
        "DSDLD language server restarted."
      );
    }
  );

  const shutdownCommand = vscode.commands.registerCommand(
    "dsdld.shutdownLanguageServer",
    async () => {
      const stopped = await runSerialLifecycleStep(async () =>
        stopLanguageServer()
      );
      await vscode.window.showInformationMessage(
        stopped
          ? "DSDLD language server stopped."
          : "DSDLD language server was not running."
      );
    }
  );

  context.subscriptions.push(configWatcher, restartCommand, shutdownCommand);
}

async function deactivate() {
  if (configWatcher) {
    configWatcher.dispose();
    configWatcher = undefined;
  }
  await runSerialLifecycleStep(async () => stopLanguageServer());
}

module.exports = {
  activate,
  deactivate,
};
