#!/usr/bin/env bash
set -euo pipefail

runai workspace -p strophf1 exec prefix-1gpu -- \
  bash -lc "cd /home/apluser/prefix/figures && tar -czf - ." \
  | tar -xzf - -C figures
