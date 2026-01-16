#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: runai/submit_train.sh [--dry-run] <configs/*.yaml>" >&2
}

die() {
  echo "$1" >&2
  exit 2
}

dry_run=false
run_config=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry_run=true; shift ;;
    -*)
      die "Unknown option: $1"
      ;;
    *)
      run_config="$1"
      shift
      ;;
  esac
done

if [[ -z "$run_config" ]]; then
  usage
  exit 2
fi

pvc_path="/home/apluser"
repo_root="${pvc_path}/prefix"
run_id="$(basename "${run_config%.*}")"
run_slug="$(echo "${run_id}" | tr '[:upper:]_' '[:lower:]-')"
if [[ "$dry_run" == "true" ]]; then
  mode="dry-run"
  module="prefix.dry_run"
  job_name="prefix-dry-run-${run_slug}"
else
  mode="train"
  module="prefix.train"
  job_name="prefix-train-${run_slug}"
fi

inner_cmd=$(
  cat <<EOF
set -euo pipefail

cd "${repo_root}"

git remote set-url origin "https://github.com/harrison-f-stropkay/prefix.git"
git fetch origin main
git checkout main
git reset --hard origin/main

uv sync --frozen
uv pip install -e .

if [[ "${run_id}" == tiny* ]]; then
  RUN_CONFIG="${run_config}" uv run python - <<'PY'
import os
from pathlib import Path

import numpy as np
import yaml
from streaming import MDSWriter

run_config = Path(os.environ["RUN_CONFIG"])
cfg = yaml.safe_load(run_config.read_text(encoding="utf-8"))
data_dir = Path(cfg["data"]["dir"])
seq_len = int(cfg["data"]["packing"]["sequence_length"])

if data_dir.exists() and any(data_dir.iterdir()):
    print(f"[run] data dir already populated: {data_dir}")
else:
    data_dir.mkdir(parents=True, exist_ok=True)
    num_samples = 256
    with MDSWriter(out=str(data_dir), columns={"input_ids": f"ndarray:int32:{seq_len}"}) as writer:
        for i in range(num_samples):
            data = (np.arange(seq_len, dtype=np.int32) + i) % 32000
            writer.write({"input_ids": data})
    print(f"[run] wrote {num_samples} fake sequences to {data_dir}")
PY
fi

echo "[run] starting ${mode}"
uv run python -m "${module}" \
  --run-config "${run_config}" \
  2>&1
EOF
)

exec runai training standard submit "$job_name" \
  --project "strophf1" \
  --image "docker-public-local.artifactory.jhuapl.edu/itsdai/runai/idp-fips-ngc2505pytorch:0.1" \
  --gpu-devices-request 8 \
  --cpu-core-request 4 \
  --cpu-memory-request 20G \
  --existing-pvc "claimname=prefix-data-5tib-project-ka8vj,path=/home/apluser" \
  --node-pools dgx-h100-80gb \
  --node-pools dgx-h100-80gb-alt \
  --node-pools dgx-h100-80gb-alt2 \
  --run-as-user \
  --environment "HOME=${pvc_path}" \
  --environment "USER=apluser" \
  --command -- bash -lc "$inner_cmd"
