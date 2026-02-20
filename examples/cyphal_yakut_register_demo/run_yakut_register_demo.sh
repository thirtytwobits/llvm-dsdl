#!/usr/bin/env sh

set -eu

: "${NODE_EXE:?NODE_EXE is required}"
: "${LOG_DIR:?LOG_DIR is required}"
: "${CYPHAL_PATH_ROOT:?CYPHAL_PATH_ROOT is required}"
: "${NODE_ID:?NODE_ID is required}"
: "${YAKUT_NODE_ID:?YAKUT_NODE_ID is required}"
: "${IFACE:?IFACE is required}"
: "${DURATION_SECONDS:?DURATION_SECONDS is required}"

YAKUT_BIN="${YAKUT_BIN:-yakut}"
NODE_BACKEND="${NODE_BACKEND:-native}"
VERIFY_HEARTBEAT="${VERIFY_HEARTBEAT:-0}"
HEARTBEAT_COUNT="${HEARTBEAT_COUNT:-3}"
HEARTBEAT_TIMEOUT_SECONDS="${HEARTBEAT_TIMEOUT_SECONDS:-12}"

case "${NODE_BACKEND}" in
  native|go)
    ;;
  *)
    echo "Invalid NODE_BACKEND: ${NODE_BACKEND} (expected: native or go)"
    exit 1
    ;;
esac

if ! command -v "${YAKUT_BIN}" >/dev/null 2>&1; then
  echo "yakut is not available (tried '${YAKUT_BIN}'). Install with: pip install yakut"
  exit 1
fi

mkdir -p "${LOG_DIR}"
NODE_LOG="${LOG_DIR}/${NODE_BACKEND}-node.log"
YAKUT_LOG="${LOG_DIR}/yakut-${NODE_BACKEND}.log"
HEARTBEAT_LOG="${LOG_DIR}/yakut-heartbeat-${NODE_BACKEND}.log"

rm -f "${NODE_LOG}" "${YAKUT_LOG}" "${HEARTBEAT_LOG}"

PID_NODE=0

cleanup() {
  if [ "${PID_NODE}" -gt 0 ]; then
    kill -INT "${PID_NODE}" 2>/dev/null || true
    wait "${PID_NODE}" 2>/dev/null || true
    PID_NODE=0
  fi
}

trap cleanup EXIT INT TERM

"${NODE_EXE}" \
  --name "${NODE_BACKEND}" \
  --node-id "${NODE_ID}" \
  --iface "${IFACE}" \
  --heartbeat-rate-hz 1 >"${NODE_LOG}" 2>&1 &
PID_NODE=$!

sleep 1

export CYPHAL_PATH="${CYPHAL_PATH_ROOT}"
export UAVCAN__UDP__IFACE="${IFACE}"
export UAVCAN__NODE__ID="${YAKUT_NODE_ID}"

run_cmd() {
  echo "\$ $*" >>"${YAKUT_LOG}"
  "$@" >>"${YAKUT_LOG}" 2>&1
  echo "" >>"${YAKUT_LOG}"
}

run_cmd "${YAKUT_BIN}" register-list "${NODE_ID}"
run_cmd "${YAKUT_BIN}" register-access "${NODE_ID}" demo.rate_hz
run_cmd "${YAKUT_BIN}" register-access "${NODE_ID}" demo.rate_hz 10
run_cmd "${YAKUT_BIN}" register-access "${NODE_ID}" demo.rate_hz
run_cmd "${YAKUT_BIN}" register-access "${NODE_ID}" uavcan.node.description "yakut configured description"
run_cmd "${YAKUT_BIN}" register-access "${NODE_ID}" uavcan.node.description
run_cmd "${YAKUT_BIN}" register-access "${NODE_ID}" sys.version
run_cmd "${YAKUT_BIN}" register-access "${NODE_ID}" sys.version "attempted overwrite"
run_cmd "${YAKUT_BIN}" register-access "${NODE_ID}" sys.version

if [ "${VERIFY_HEARTBEAT}" = "1" ]; then
  echo "\$ ${YAKUT_BIN} subscribe -N ${HEARTBEAT_COUNT} uavcan.node.heartbeat +M" >>"${YAKUT_LOG}"
  "${YAKUT_BIN}" subscribe -N "${HEARTBEAT_COUNT}" uavcan.node.heartbeat +M >"${HEARTBEAT_LOG}" 2>&1 &
  PID_SUB=$!
  START_SECONDS="$(date +%s)"
  while kill -0 "${PID_SUB}" 2>/dev/null; do
    NOW_SECONDS="$(date +%s)"
    ELAPSED_SECONDS=$((NOW_SECONDS - START_SECONDS))
    if [ "${ELAPSED_SECONDS}" -ge "${HEARTBEAT_TIMEOUT_SECONDS}" ]; then
      kill -TERM "${PID_SUB}" 2>/dev/null || true
      wait "${PID_SUB}" 2>/dev/null || true
      echo "heartbeat subscribe timed out after ${HEARTBEAT_TIMEOUT_SECONDS}s" >>"${YAKUT_LOG}"
      echo "" >>"${YAKUT_LOG}"
      echo "ERROR: heartbeat subscribe timed out after ${HEARTBEAT_TIMEOUT_SECONDS}s"
      exit 1
    fi
    sleep 1
  done
  wait "${PID_SUB}"
  echo "" >>"${YAKUT_LOG}"

  if ! grep -Eq "\"source_node_id\"[[:space:]]*:[[:space:]]*${NODE_ID}" "${HEARTBEAT_LOG}"; then
    echo "ERROR: heartbeat verification failed: expected source_node_id ${NODE_ID}"
    echo "--- Heartbeat log ---"
    sed -n '1,120p' "${HEARTBEAT_LOG}" || true
    exit 1
  fi
fi

sleep "${DURATION_SECONDS}"

cleanup

echo "=== yakut register demo summary ==="
echo "backend: ${NODE_BACKEND}"
echo "node executable: ${NODE_EXE}"
echo "node id: ${NODE_ID}"
echo "yakut node id: ${YAKUT_NODE_ID}"
echo "iface: ${IFACE}"
echo "logs: ${LOG_DIR}"
echo ""
echo "--- ${NODE_BACKEND} node log (first 60 lines) ---"
sed -n '1,60p' "${NODE_LOG}" || true
echo ""
echo "--- Yakut command log (first 120 lines) ---"
sed -n '1,120p' "${YAKUT_LOG}" || true
if [ "${VERIFY_HEARTBEAT}" = "1" ]; then
  echo ""
  echo "--- Yakut heartbeat log (first 120 lines) ---"
  sed -n '1,120p' "${HEARTBEAT_LOG}" || true
fi
