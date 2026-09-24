#!/usr/bin/env bash
# Usage: build_engines.sh [-o OUTDIR] [-w WORKSPACE_MB] ONNX... [-- TRTEXEC_ARGS...]
set -euo pipefail

DEPLOY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="$DEPLOY_ROOT/../artifacts/engines"
WS=2048
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

while getopts ":o:w:h" opt; do
  case "$opt" in
    o) OUTDIR="$OPTARG" ;;
    w) WS="$OPTARG" ;;
    h) sed -n 2p "$0"; exit 0 ;;
    *) sed -n 2p "$0" >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))

ONNX=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do ONNX+=("$1"); shift; done
[ $# -gt 0 ] && shift
EXTRA=("$@")

[ ${#ONNX[@]} -gt 0 ] || { sed -n 2p "$0" >&2; exit 2; }
[ -x "$TRTEXEC" ] || command -v "$TRTEXEC" >/dev/null 2>&1 || {
  echo "trtexec not found at $TRTEXEC; set TRTEXEC=/path/to/trtexec" >&2; exit 1; }
mkdir -p "$OUTDIR"

for onnx in "${ONNX[@]}"; do
  [ -f "$onnx" ] || { echo "no such file: $onnx" >&2; exit 1; }
  name=$(basename "$onnx" .onnx)
  echo "=== $onnx -> $OUTDIR/$name.plan ==="
  "$TRTEXEC" \
    --onnx="$onnx" \
    --saveEngine="$OUTDIR/$name.plan" \
    --stronglyTyped \
    --memPoolSize=workspace:"$WS" \
    --builderOptimizationLevel=3 \
    --timingCacheFile="$OUTDIR/timing.cache" \
    --profilingVerbosity=detailed \
    --skipInference \
    "${EXTRA[@]}" 2>&1 | tee "$OUTDIR/build_$name.log"
done
