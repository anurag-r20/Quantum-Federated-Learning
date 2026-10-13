#!/usr/bin/env python3
"""Check that a test suite notices deliberate bugs ("mutations") in the code it tests.

Copies the repository once to a temporary folder, then for each mutation: applies one exact text replacement,
runs pytest, records whether the suite failed (the mutation was caught), and restores the file. The original
repository is never modified.

Mutations file (JSON):
    [{"file": "src/pkg/model.py", "find": "+= w/2", "replace": "+= w", "description": "Q off-diagonal w/2 -> w"}]
`find` must occur exactly once in the file (add surrounding text to make it unique), or the mutation is
reported as "not applied". Use "count": "all" to replace every occurrence instead.

Usage:
    python mutation_check.py --repo . --mutations mutations.json --python /path/to/env/python \
        --pytest-args "-m 'not integration' -x -q -p no:cacheprovider"

Exit code: 0 if every applied mutation was caught, 1 if any survived or was not applied, 2 on usage errors.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

IGNORE = shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".venv", "venv", "node_modules",
                                ".ipynb_checkpoints", "htmlcov", ".coverage", "*.pyc")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a test suite against deliberate mutations of the code.")
    parser.add_argument("--repo", default=".", help="repository root (default: current directory)")
    parser.add_argument("--mutations", required=True, help="JSON file with the list of mutations")
    parser.add_argument("--python", default=sys.executable, help="python that has pytest and the repo's dependencies")
    parser.add_argument("--pytest-args", default="-x -q -p no:cacheprovider", help="arguments passed to pytest")
    parser.add_argument("--timeout", type=int, default=900, help="seconds allowed per pytest run")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    try:
        mutations = json.loads(Path(args.mutations).read_text())
    except (OSError, json.JSONDecodeError) as error:
        print(f"Cannot read mutations: {error}")
        return 2
    command = [args.python, "-m", "pytest", *shlex.split(args.pytest_args)]

    with tempfile.TemporaryDirectory(prefix="mutation-check-") as tmp:
        work = Path(tmp) / repo.name
        shutil.copytree(repo, work, ignore=IGNORE)

        baseline = subprocess.run(command, cwd=work, capture_output=True, text=True, timeout=args.timeout)
        if baseline.returncode != 0:
            print("The unmutated suite already fails; fix that first. Last lines:")
            print("\n".join(baseline.stdout.strip().splitlines()[-5:]))
            return 2
        print(f"baseline: {baseline.stdout.strip().splitlines()[-1]}")

        survived = 0
        for number, mutation in enumerate(mutations, start=1):
            target = work / mutation["file"]
            label = mutation.get("description") or f"{mutation['find']!r} -> {mutation['replace']!r}"
            original = target.read_text()
            occurrences = original.count(mutation["find"])
            replace_all = mutation.get("count") == "all"
            if occurrences == 0 or (occurrences > 1 and not replace_all):
                print(f"[{number}] NOT APPLIED  {label}  ('find' occurs {occurrences} times in {mutation['file']})")
                survived += 1
                continue
            target.write_text(original.replace(mutation["find"], mutation["replace"]))
            try:
                run = subprocess.run(command, cwd=work, capture_output=True, text=True, timeout=args.timeout)
                caught = run.returncode != 0
                summary = run.stdout.strip().splitlines()[-1] if run.stdout.strip() else ""
            except subprocess.TimeoutExpired:
                caught, summary = True, "timed out (counted as caught)"
            finally:
                target.write_text(original)
            survived += not caught
            print(f"[{number}] {'CAUGHT  ' if caught else 'SURVIVED'}  {label}  | {summary}")

    print(f"{len(mutations) - survived}/{len(mutations)} mutations caught")
    return 0 if survived == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
