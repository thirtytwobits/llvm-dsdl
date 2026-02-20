#!/usr/bin/env bash
set -euo pipefail

need_llvm_mlir=0
need_zstd=0

if ! [ -f /usr/lib/llvm-19/lib/cmake/mlir/MLIRConfig.cmake ]; then
  need_llvm_mlir=1
fi

if ! find /usr/lib -path "*/cmake/zstd/zstdConfig.cmake" -print -quit | grep -q .; then
  need_zstd=1
fi

if [ "$need_llvm_mlir" -eq 1 ] || [ "$need_zstd" -eq 1 ]; then
  if command -v sudo >/dev/null 2>&1 && [ "$(id -u)" -ne 0 ]; then
    sudo apt-get update
    sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libzstd-dev \
      libmlir-19-dev \
      mlir-19-tools
  else
    apt-get update
    env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libzstd-dev \
      libmlir-19-dev \
      mlir-19-tools
  fi
fi

if ! command -v lit >/dev/null 2>&1 && ! command -v llvm-lit >/dev/null 2>&1; then
  python3 -m pip install --break-system-packages lit
fi

git submodule update --init --recursive
cmake --preset dev-llvm-env
