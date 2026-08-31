#!/usr/bin/env bash
# Build a target-specific TensorRT plan from a developer-supplied ONNX graph.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 MODEL.onnx OUT.plan [trtexec profile/other options]" >&2
  exit 2
fi

onnx=$1
plan=$2
shift 2
if [[ ! -f $onnx ]]; then
  echo "missing ONNX: $onnx" >&2
  exit 1
fi

# Precision is encoded in the ONNX graph. This wrapper never quantizes or
# promotes the graph to another precision. --noTF32 is allowed for FP32 math.
for option in "$@"; do
  case "$option" in
    --fp16|--int8|--fp8|--bf16|--best)
      echo "precision conversion is outside this repo; supply a prepared ONNX graph" >&2
      exit 2 ;;
  esac
done

trtexec=${TRTEXEC:-trtexec}
workspace=${TRT_WORKSPACE_MIB:-2048}
if [[ ! $workspace =~ ^[1-9][0-9]*$ ]]; then
  echo "TRT_WORKSPACE_MIB must be a positive integer" >&2
  exit 2
fi
mkdir -p "$(dirname "$plan")"
"$trtexec" --onnx="$onnx" --saveEngine="$plan" --stronglyTyped \
  --memPoolSize="workspace:$workspace" --builderOptimizationLevel=3 \
  --skipInference "$@"
