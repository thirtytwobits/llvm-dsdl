#!/usr/bin/env sh

set -eu

: "${NODE_EXE:?NODE_EXE is required}"
: "${LOG_DIR:?LOG_DIR is required}"
: "${DURATION_SECONDS:?DURATION_SECONDS is required}"
: "${PERIOD_MS:?PERIOD_MS is required}"

mkdir -p "${LOG_DIR}"

A_LOG="${LOG_DIR}/a.log"
B_LOG="${LOG_DIR}/b.log"

rm -f "${A_LOG}" "${B_LOG}"

PID_A=0
PID_B=0

cleanup() {
  if [ "${PID_A}" -gt 0 ]; then
    kill -INT "${PID_A}" 2>/dev/null || true
    wait "${PID_A}" 2>/dev/null || true
    PID_A=0
  fi
  if [ "${PID_B}" -gt 0 ]; then
    kill -INT "${PID_B}" 2>/dev/null || true
    wait "${PID_B}" 2>/dev/null || true
    PID_B=0
  fi
}

trap cleanup EXIT INT TERM

"${NODE_EXE}" \
  --name A \
  --node-id 42 \
  --peer-node-id 43 \
  --service-id 300 \
  --iface 127.0.0.1 \
  --period-ms "${PERIOD_MS}" >"${A_LOG}" 2>&1 &
PID_A=$!

"${NODE_EXE}" \
  --name B \
  --node-id 43 \
  --peer-node-id 42 \
  --service-id 300 \
  --iface 127.0.0.1 \
  --period-ms "${PERIOD_MS}" >"${B_LOG}" 2>&1 &
PID_B=$!

sleep "${DURATION_SECONDS}"

cleanup

echo "=== cyphal ping-pong demo summary ==="
echo "node: ${NODE_EXE}"
echo "duration_seconds: ${DURATION_SECONDS}"
echo "period_ms: ${PERIOD_MS}"
echo "logs: ${LOG_DIR}"
echo ""
echo "--- A (first 40 lines) ---"
sed -n '1,40p' "${A_LOG}" || true
echo ""
echo "--- B (first 40 lines) ---"
sed -n '1,40p' "${B_LOG}" || true
