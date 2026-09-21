#!/usr/bin/env python3
"""Fail if any requirements*.txt line isn't pinned to an exact version (==)."""

import re
import sys
from pathlib import Path

PINNED = re.compile(r"^[A-Za-z0-9_.-]+(\[[A-Za-z0-9_,-]+\])?==[A-Za-z0-9_.\-+]+$")


def find_violations(path: Path) -> list[str]:
    violations = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if not PINNED.match(line):
            violations.append(raw_line)
    return violations


def main() -> int:
    files = sorted(Path(".").glob("requirements*.txt"))
    if not files:
        print("No requirements*.txt files found.")
        return 1

    exit_code = 0
    for path in files:
        violations = find_violations(path)
        if violations:
            exit_code = 1
            print(f"{path}: found unpinned/invalid dependency line(s):")
            for v in violations:
                print(f"  - {v!r} (expected an exact '==' pin, e.g. package==1.2.3)")
        else:
            print(f"{path}: all dependencies pinned.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
