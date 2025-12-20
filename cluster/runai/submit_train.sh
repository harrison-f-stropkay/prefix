#!/usr/bin/env bash
set -euo pipefail

run_config="${1:-}"
if [[ -z "$run_config" ]]; then
  echo "Usage: cluster/runai/submit_train.sh <configs/runs/*.toml>" >&2
  exit 2
fi

project="strophf1"
image="docker-public-local.artifactory.jhuapl.edu/itsdai/runai/idp-fips-ngc2505pytorch:0.1"
pvc_claim="prefix-data-project-zhncd"
pvc_path="/home/apluser"
repo_url="https://gitlab.jhuapl.edu/strophf1/prefix.git"

job_name="prefix-train-$(basename "${run_config%.*}")"

inner_cmd=$(
  cat <<EOF
set -euo pipefail

rm -rf /tmp/prefix
git clone "${repo_url}" /tmp/prefix
cd /tmp/prefix
git checkout main

RUN_CONFIG_PATH="${run_config}"
RUN_NAME="\$(python - "\$RUN_CONFIG_PATH" <<'PY'
import sys
import tomllib
from pathlib import Path

path = Path(sys.argv[1])
data = tomllib.loads(path.read_text())
print(data["run"]["name"])
PY
)"

run_dir="${pvc_path}/runs/\${RUN_NAME}"
log_dir="\${run_dir}/logs"
meta_dir="\${run_dir}/meta"
mkdir -p "\${log_dir}" "\${meta_dir}/configs"

git_commit="\$(git rev-parse HEAD)"
echo "git_commit=\${git_commit}" > "\${meta_dir}/git.txt"

uv sync --frozen

cp -f "\$RUN_CONFIG_PATH" "\${meta_dir}/configs/run.toml"
run_spec_json="\${meta_dir}/configs/run_spec.json"
python - "\$RUN_CONFIG_PATH" "\${meta_dir}/configs" "\$run_spec_json" <<'PY'
import json
import shutil
import sys
import tomllib
from pathlib import Path

run_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
run_spec_path = Path(sys.argv[3])

data = tomllib.loads(run_path.read_text())
run_spec_path.write_text(json.dumps(data, indent=2, sort_keys=True))

configs = data.get("configs") or {}

def copy(key, name):
    src = configs.get(key)
    if src:
        shutil.copyfile(src, out_dir / f"{name}.yaml")

copy("objective", "objective")
copy("data", "data")
copy("model", "model")
copy("train", "train")
copy("eval", "eval")
PY

echo "[run] starting training"
echo "[run] git_commit=\${git_commit}"
set -x
uv run python -m prefix.train \\
  --output-dir "\${run_dir}" \\
  --run-spec "\${run_spec_json}" \\
  2>&1 | tee "\${log_dir}/train.log"
EOF
)

exec runai training standard submit "$job_name" \
  --project "$project" \
  --image "$image" \
  --preemptible \
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
