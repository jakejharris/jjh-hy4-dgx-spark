#!/usr/bin/env bash
# Run the full-precision 78-layer BF16 teacher control and panel evaluation in a container.
# Required env: SOURCE_URL (HTTP shard server), IMAGE (PyTorch container), SITE_DIR (transformers).
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs artifacts
exec 9>artifacts/runner.lock
flock -n 9

SOURCE_URL="${SOURCE_URL:?Set SOURCE_URL to the read-only HTTP BF16 shard server}"
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
SITE_DIR="${SITE_DIR:?Set SITE_DIR to the staged transformers site-packages directory}"

if [[ -f SHA256SUMS ]]; then
  sha256sum -c SHA256SUMS
fi

stamp=$(date -u +%Y%m%dT%H%M%SZ)

run_step() {
  local step=$1
  local name="hy4-eval-$step-$stamp"
  docker run --rm --name "$name" --pull never --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit nofile=1048576 \
    -v "$PWD:/s" -v "$SITE_DIR:/site:ro" -w /s "$IMAGE" \
    env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE PYTHONPATH=/site PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    python /s/teacher.py "$step" --source "$SOURCE_URL" --out /s/artifacts \
    2>&1 | tee "logs/$stamp.$step.log"
}

if [[ ! -f artifacts/control.json ]]; then
  run_step control
fi
run_step teacher
