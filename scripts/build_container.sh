#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

if [[ -f "$AI4PH_IMAGE" ]]; then
    echo "container already exists: $AI4PH_IMAGE"
    exit 0
fi

build_dir="$(mktemp -d "${TMPDIR:-/scratch}/ai4ph-build.XXXXXX")"
trap 'rm -rf "$build_dir"' EXIT
tmp_image="$build_dir/ai4ph-openrlhf-0.11.2.sif"

apptainer build "$tmp_image" "$AI4PH_ROOT/containers/openrlhf.def"
mv "$tmp_image" "$AI4PH_IMAGE"
apptainer inspect "$AI4PH_IMAGE"
echo "container ready: $AI4PH_IMAGE"

