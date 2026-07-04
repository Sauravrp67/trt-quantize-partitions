# lock_clocks.sh — pin GPU clocks for reproducible latency/power (needs sudo on some systems)
#!/usr/bin/env bash
set -euo pipefail
GCLK="${1:-1500}"   # MHz; pick a stable value below boost
nvidia-smi -lgc "${GCLK},${GCLK}" || echo "warn: could not lock clocks (laptop GPU may disallow)"