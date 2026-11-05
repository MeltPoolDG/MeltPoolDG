#!/usr/bin/env bash
set -euo pipefail

root="${1:-applications}"
pairs_file="$(mktemp)"
trap 'rm -f "$pairs_file"' EXIT

for app_dir in "$root"/mp-*; do
  [ -d "$app_dir" ] || continue

  schema_file="$app_dir/parameters.schema.json"
  [ -f "$schema_file" ] || continue

  find "$app_dir" \
    -type f \
    -name "*.json" \
    ! -name "parameters.schema.json" \
    -printf '%p\t'"$schema_file"'\n'
done > "$pairs_file"

python3 scripts/parameters/validate_schema.py --pairs-file "$pairs_file" \
  --exclude-category "case specific" 
