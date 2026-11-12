#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/env.sh"

"$script_dir/container_exec.sh" python - <<'PY'
from huggingface_hub import snapshot_download

for model_id in ("Qwen/Qwen2.5-32B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"):
    path = snapshot_download(repo_id=model_id)
    print(f"ready: {model_id} -> {path}", flush=True)
PY

