#!/usr/bin/env bash
set -euo pipefail

workspace="prefix-1gpu"
project="strophf1"

runai workspace resu@e "${workspace}" -p "${project}" >/dev/null 2>&1 || true

for _ in {1..60}; do
  status=$(runai workspace list -p "${project}" | awk -v w="${workspace}" '$1==w {print $4}')
  if [[ "${status}" == "Running" ]]; then
    break
  fi
  sleep 10
done

runai workspace -p "${project}" exec "${workspace}" -- \
  bash -lc "cd /home/apluser/prefix && git pull && uv run python scripts/charbench_guess_breakdown.py && uv run python scripts/example_target_dists.py && uv run python scripts/plot_prefix_counts.py && uv run python scripts/plot_metrics.py"

bash "$(dirname "$0")/get_figures.sh"
 