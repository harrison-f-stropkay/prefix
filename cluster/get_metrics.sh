#!/usr/bin/env bash
set -euo pipefail

# Stream metrics without corrupting the tar stream.
# Run:ai merges stderr into stdout, and tar warns if files change mid-read.
# Snapshot each metrics.jsonl into a temp dir, then stream it per-run.
# This avoids truncated gzip streams when files are actively being appended.
for d in ce_seed_0 label_smoothing_seed_0 prefix_norm_eps0p1_seed0 \
         prefix_simple_seed_0 prefix_unnorm_eps0p1_seed0; do
  runai -q workspace -p strophf1 exec prefix-cpu-only -- \
    bash -lc "set -euo pipefail; cd /home/apluser/prefix/runs; \
    tmp=\$(mktemp -d); mkdir -p \"\$tmp/$d\"; cp $d/metrics.jsonl \"\$tmp/$d/metrics.jsonl\"; \
    tar -czf - -C \"\$tmp\" $d; rm -rf \"\$tmp\"" \
    | tar -xzf - -C runs
done
