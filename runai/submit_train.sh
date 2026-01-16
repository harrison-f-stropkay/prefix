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

project="strophf1"
image="docker-public-local.artifactory.jhuapl.edu/itsdai/runai/idp-fips-ngc2505pytorch:0.1"
pvc_claim="prefix-data-project-zhncd"
pvc_path="/home/apluser"
repo_root="${pvc_path}/prefix"
run_id="$(basename "${run_config%.*}")"
run_slug="$(echo "${run_id}" | tr '[:upper:]_' '[:lower:]-')"
if [[ "$dry_run" == "true" ]]; then
  mode="dry-run"
  module="prefix.dry_run"
  log_name="dry_run.log"
  runs_root="${repo_root}/dry_runs"
  job_name="prefix-dry-run-${run_slug}"
else
  mode="train"
  module="prefix.train"
  log_name="train.log"
  runs_root="${repo_root}/runs"
  job_name="prefix-train-${run_slug}"
fi

inner_cmd=$(
  cat <<EOF
set -euo pipefail

cd "${repo_root}"

git fetch origin main
git checkout main
git reset --hard origin/main

uv sync --frozen

run_name="\$(uv run python - "${run_config}" <<'PY'
import sys
from pathlib import Path

from prefix.config import load_run_config

path = Path(sys.argv[1])
data = load_run_config(path)
print(data["run"]["name"])
PY
)"

run_dir="${runs_root}/\${run_name}"
log_dir="\${run_dir}/logs"
meta_dir="\${run_dir}/meta"
mkdir -p "\${log_dir}" "\${meta_dir}/configs"

git fetch origin main
git checkout main
git reset --hard origin/main
git_commit="\$(git rev-parse HEAD)"
echo "git_commit=\${git_commit}" > "\${meta_dir}/git.txt"

cp -f "${run_config}" "\${meta_dir}/configs/run.yaml"
run_config_json="\${meta_dir}/configs/run_config.json"
uv run python - "${run_config}" "\${run_config_json}" <<'PY'
import json
import sys
from pathlib import Path

from prefix.config import load_run_config

run_path = Path(sys.argv[1])
run_config_path = Path(sys.argv[2])

data = load_run_config(run_path)
run_config_path.write_text(json.dumps(data, indent=2, sort_keys=True))
PY

echo "[run] starting ${mode}"
echo "[run] git_commit=\${git_commit}"
set -x
uv run python -m "${module}" \\
  --output-dir "\${run_dir}" \\
  --run-config "\${run_config_json}" \\
  2>&1 | tee "\${log_dir}/${log_name}"
EOF
)

exec runai training standard submit "$job_name" \
  --project "${project}" \
  --image "${image}" \
  --gpu-devices-request 8 \
  --cpu-core-request 4 \
  --cpu-memory-request 20G \
  --existing-pvc "claimname=${pvc_claim},path=${pvc_path}" \
  --node-pools dgx-h100-80gb \
  --node-pools dgx-h100-80gb-alt \
  --node-pools dgx-h100-80gb-alt2 \
  --run-as-user \
  --environment "HOME=${pvc_path}" \
  --environment "USER=apluser" \
  --command -- bash -lc "$inner_cmd"
