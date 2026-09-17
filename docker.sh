#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image="${FND_DOCKER_IMAGE:-fnd_flow:release}"

if ! docker image inspect "${image}" >/dev/null 2>&1; then
  echo "Docker image not found: ${image}" >&2
  echo "Build Dockerfile.release as described in README.md, or set FND_DOCKER_IMAGE to a published image." >&2
  exit 1
fi

tty_args=()
if [[ -t 0 && -t 1 ]]; then
  tty_args=(-it)
fi

if [[ $# -eq 0 ]]; then
  set -- /bin/bash
fi

source_root="${FND_SOURCE_ROOT:-$(dirname "${root_dir}")/FND}"
source_mount=()
if [[ -d "${source_root}" && "$(realpath "${source_root}")" != "$(realpath "${root_dir}")" ]]; then
  source_mount=(-v "${source_root}:/source:ro")
fi

exec docker run --rm "${tty_args[@]}" \
  --ipc=host \
  --gpus all \
  -u "$(id -u):$(id -g)" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -e HOME=/tmp \
  -e XDG_CONFIG_HOME=/tmp/fnd_config \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "${root_dir}:/workspace" \
  "${source_mount[@]}" \
  -w /workspace \
  "${image}" "$@"
