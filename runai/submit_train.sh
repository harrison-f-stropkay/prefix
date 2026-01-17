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
  # Use torchrun so dry runs exercise multi-rank DDP behavior.
  launcher="torchrun --nproc_per_node=8"
else
  mode="train"
  module="prefix.train"
  job_name="prefix-train-${run_slug}"
  launcher="python"
fi

inner_cmd="set -euo pipefail; \
mkdir -p \"${repo_root}/runs/${run_id}/logs\"; \
: > \"${repo_root}/runs/${run_id}/logs/${mode}.log\"; \
exec >> \"${repo_root}/runs/${run_id}/logs/${mode}.log\" 2>&1; \
cd \"${repo_root}\"; \
git remote set-url origin \"https://github.com/harrison-f-stropkay/prefix.git\"; \
git fetch origin main; \
git checkout main; \
git reset --hard origin/main; \
uv sync --frozen; \
uv pip install -e .; \
if [[ \"${run_id}\" == tiny* || \"${run_id}\" == *smoke* ]]; then uv run python scripts/make_fake_mds.py --run-config \"${run_config}\"; fi; \
echo \"[run] starting ${mode}\"; \
set +e; \
uv run ${launcher} -m \"${module}\" --run-config \"${run_config}\"; \
status=$?; \
set -e; \
exit ${status}"

exec runai training standard submit "$job_name" \
  --project "strophf1" \
  --image "docker-public-local.artifactory.jhuapl.edu/itsdai/runai/idp-fips-ngc2505pytorch:0.1" \
  --gpu-devices-request 8 \
  --cpu-core-request 16 \
  --cpu-memory-request 64G \
  --existing-pvc "claimname=prefix-data-5tib-project-ka8vj,path=/home/apluser" \
  --node-pools dgx-h100-80gb \
  --node-pools dgx-h100-80gb-alt \
  --node-pools dgx-h100-80gb-alt2 \
  --run-as-user \
  --environment "HOME=${pvc_path}" \
  --environment "USER=apluser" \
  --command -- bash -lc "$inner_cmd"
