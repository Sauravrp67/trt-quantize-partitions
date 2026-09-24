#!/usr/bin/env bash
# Usage: bash deploy/setup.sh [--eval]
set -euo pipefail
DEPLOY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${JETSON_PYTHON:-/usr/bin/python3}"
VENV_DIR="${JETSON_VENV:-$DEPLOY_ROOT/../.venv-jetson}"
REQUIREMENTS="$DEPLOY_ROOT/requirements.txt"
case "${1:-}" in
  --eval) REQUIREMENTS="$DEPLOY_ROOT/requirements-eval.txt" ;;
  -h|--help) sed -n 2p "$0"; exit 0 ;;
  "") ;;
  *) sed -n 2p "$0" >&2; exit 2 ;;
esac
[ "$#" -le 1 ] || { sed -n 2p "$0" >&2; exit 2; }
"$PYTHON_BIN" -c 'import tensorrt, cv2; print("TensorRT", tensorrt.__version__, "OpenCV", cv2.__version__)' || {
  echo "Use JETSON_PYTHON=/path/to/python with the board's TensorRT and OpenCV bindings." >&2
  exit 1
}
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
[ -x "$TRTEXEC" ] || command -v "$TRTEXEC" >/dev/null 2>&1 || {
  echo "trtexec not found; set TRTEXEC=/path/to/trtexec" >&2; exit 1;
}
"$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install -r "$REQUIREMENTS"
"$VENV_DIR/bin/python" -c 'import tensorrt, cv2, numpy, PIL, polygraphy, yaml; print("Jetson runtime ready")'
printf 'Activate with: source %q/bin/activate\n' "$VENV_DIR"
