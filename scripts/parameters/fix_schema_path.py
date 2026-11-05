#!/usr/bin/env python3
"""
Ensure that JSON input files of the applications reference the parameter schema
of their application via a relative "$schema" path as the first entry, e.g.,

  {
    "$schema": "../../parameters.schema.json",
    ...

Files with a missing, wrong or misplaced "$schema" entry are fixed in place.
The exit code is 1 if any file was modified, so that the pre-commit hook fails
and the fixed files can be reviewed and staged.

Usage: fix_schema_path.py <json_file> [<json_file> ...]

To process all tracked files, run: pre-commit run fix-json-schema-path --all-files
"""

from __future__ import annotations

import json
import posixpath
import sys
from pathlib import Path

SCHEMA_KEY = "$schema"
SCHEMA_NAME = "parameters.schema.json"


def expected_schema_path(json_file: Path) -> str | None:
    """Return the relative path from json_file to the schema of its application."""
    parts = json_file.parts
    if len(parts) < 3 or parts[0] != "applications" or not parts[1].startswith("mp-"):
        return None

    schema_file = Path(parts[0], parts[1], SCHEMA_NAME)
    if not schema_file.is_file():
        return None

    return posixpath.relpath(schema_file.as_posix(), json_file.parent.as_posix())


def fix_file(json_file: Path) -> bool:
    """Fix the "$schema" entry of json_file. Return True if the file was modified."""
    expected = expected_schema_path(json_file)
    if expected is None:
        return False

    data = json.loads(json_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return False

    if next(iter(data), None) == SCHEMA_KEY and data[SCHEMA_KEY] == expected:
        return False

    old = data.pop(SCHEMA_KEY, None)
    fixed = {SCHEMA_KEY: expected, **data}
    json_file.write_text(json.dumps(fixed, indent=2) + "\n", encoding="utf-8")

    if old is None:
        print(f"{json_file}: added \"{SCHEMA_KEY}\": \"{expected}\"")
    elif old != expected:
        print(f"{json_file}: changed \"{SCHEMA_KEY}\" from \"{old}\" to \"{expected}\"")
    else:
        print(f"{json_file}: moved \"{SCHEMA_KEY}\" to the first entry")
    return True


def main() -> int:
    modified = False
    for name in sys.argv[1:]:
        json_file = Path(name)
        if json_file.name == SCHEMA_NAME:
            continue
        try:
            modified |= fix_file(json_file)
        except json.JSONDecodeError as exc:
            print(f"{json_file}: invalid JSON ({exc})", file=sys.stderr)
            modified = True
    return 1 if modified else 0


if __name__ == "__main__":
    raise SystemExit(main())
