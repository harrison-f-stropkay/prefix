#!/usr/bin/env bash
set -euo pipefail

# Stream metrics without corrupting the tar stream.
# Run:ai merges stderr into stdout, and tar warns if files change mid-read.
# Snapshot each metrics.jsonl into a temp dir, then stream it in one tar.
# This avoids truncated gzip streams when files are actively being appended.
# NOTE: do not use `-q` here; it suppresses stdout and breaks the tar stream.
runai workspace -p strophf1 exec prefix-cpu-only -- \
  bash -lc "set -euo pipefail; cd /home/apluser/prefix/runs; \
  tmp=\$(mktemp -d); \
  for f in */metrics.jsonl; do \
    [ -f \"\$f\" ] || continue; \
    d=\$(dirname \"\$f\"); \
    mkdir -p \"\$tmp/\$d\"; \
    cp \"\$f\" \"\$tmp/\$d/metrics.jsonl\"; \
  done; \
  tar -czf - -C \"\$tmp\" .; rm -rf \"\$tmp\"" \
  | tar -xzf - -C runs

uv run scripts/plot_metrics.py