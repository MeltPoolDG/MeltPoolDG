#!/usr/bin/env bash
# Validate the JSON input files of all applications against the parameter
# schema (parameters.schema.json) of the respective application.
#
# Usage: validate_schema_all.sh [--tracked-only] [root]
#   --tracked-only  Only validate files tracked by git. Used by the pre-commit
#                   hook, so that local files (e.g., simulation output) do not
#                   block a commit.
#   root            Directory containing the mp-* applications
#                   (default: applications).
set -euo pipefail

tracked_only=false
root="applications"

for arg in "$@"; do
  case "$arg" in
    --tracked-only) tracked_only=true ;;
    *) root="$arg" ;;
  esac
done

pairs_file="$(mktemp)"
trap 'rm -f "$pairs_file"' EXIT

for app_dir in "$root"/mp-*; do
  [ -d "$app_dir" ] || continue

  schema_file="$app_dir/parameters.schema.json"
  [ -f "$schema_file" ] || continue

  if $tracked_only; then
    git ls-files -z -- "$app_dir/*.json" |
      while IFS= read -r -d '' json_file; do
        [ "$(basename "$json_file")" != "parameters.schema.json" ] || continue
        [ -f "$json_file" ] || continue
        printf '%s\t%s\n' "$json_file" "$schema_file"
      done
  else
    find "$app_dir" \
      -type f \
      -name "*.json" \
      ! -name "parameters.schema.json" \
      -printf '%p\t'"$schema_file"'\n'
  fi
done > "$pairs_file"

python3 scripts/parameters/validate_schema.py --pairs-file "$pairs_file" \
  --exclude-category "case specific"
