#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

if [[ ! -f "$AI4PH_IMAGE" ]]; then
    echo "missing container: $AI4PH_IMAGE" >&2
    echo "run scripts/build_container.sh first" >&2
    exit 1
fi

gpu_args=()
if [[ -e /dev/nvidiactl ]]; then
    gpu_args+=(--nv --env "TRITON_LIBCUDA_PATH=/.singularity.d/libs")
fi

exec apptainer exec "${gpu_args[@]}" \
    --env "HF_HOME=$HF_HOME" \
    --env "RAY_TMPDIR=$RAY_TMPDIR" \
    --env "PYTHONPATH=$PYTHONPATH" \
    "$AI4PH_IMAGE" "$@"

