//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

const path = require("path");
const fs = require("fs");
const os = require("os");
const { runTests } = require("@vscode/test-electron");

async function main() {
  try {
    const extensionDevelopmentPath = path.resolve(__dirname, "..");
    const extensionTestsPath = path.resolve(__dirname, "suite", "index");
    const workspacePath = path.resolve(__dirname, "workspace");
    const shortBasePath = path.join(os.tmpdir(), "dsdld-vscode-test");
    const userDataPath = path.join(shortBasePath, "u");
    const extensionsPath = path.join(shortBasePath, "e");
    const localVSCodeExecutable =
      "/Applications/Visual Studio Code.app/Contents/MacOS/Electron";

    fs.mkdirSync(userDataPath, { recursive: true });
    fs.mkdirSync(extensionsPath, { recursive: true });

    const options = {
      extensionDevelopmentPath,
      extensionTestsPath,
      launchArgs: [
        "--disable-extensions",
        `--user-data-dir=${userDataPath}`,
        `--extensions-dir=${extensionsPath}`,
        workspacePath,
      ],
      extensionTestsEnv: {
        DSDLD_BINARY: process.env.DSDLD_BINARY || "",
      },
    };
    if (fs.existsSync(localVSCodeExecutable)) {
      options.vscodeExecutablePath = localVSCodeExecutable;
    }

    await runTests(options);
  } catch (err) {
    console.error("VSCode extension tests failed");
    console.error(err);
    process.exit(1);
  }
}

main();
