#!/usr/bin/env bash
# Source this file; paths can be overridden before sourcing.

if ! command -v ws_find >/dev/null 2>&1; then
    echo "ws_find is unavailable; run this on a Helma frontend or compute node" >&2
    return 1 2>/dev/null || exit 1
fi

export AI4PH_ROOT="${AI4PH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export AI4PH_WORKSPACE="${AI4PH_WORKSPACE:-$(ws_find ai4ph)}"
export AI4PH_IMAGE="${AI4PH_IMAGE:-$AI4PH_WORKSPACE/containers/ai4ph-openrlhf-0.11.2.sif}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$AI4PH_WORKSPACE/cache/apptainer}"
export HF_HOME="${HF_HOME:-$AI4PH_WORKSPACE/cache/huggingface}"
export RAY_TMPDIR="${RAY_TMPDIR:-${TMPDIR:-/scratch}/ai4ph-ray}"
export PYTHONPATH="$AI4PH_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$(dirname "$AI4PH_IMAGE")" "$APPTAINER_CACHEDIR" "$HF_HOME" "$RAY_TMPDIR"

