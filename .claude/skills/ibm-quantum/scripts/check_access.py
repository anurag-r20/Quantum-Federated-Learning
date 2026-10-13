#!/usr/bin/env python3
"""Check IBM Quantum access from this machine and report what it can use. Spends no QPU time.

Prints the saved accounts (the API key is never printed), connects with one of them, and reports the
active instance, its usage, and the backends it can reach.

Exit codes: 0 = access works; 2 = no saved account (run save_account.py); 1 = a saved account fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings

# Known IAM / platform errors and what to do about them, matched against the error text.
DIAGNOSES = [
    ("blocked",
     "IBM Cloud reports the account this API key belongs to as BLOCKED. Log in at https://quantum.cloud.ibm.com "
     "(or https://cloud.ibm.com) to see why, for example an expired trial or a verification or billing hold. Then "
     "create a new API key in an active account and run save_account.py again."),
    ("BXNIM0415E",
     "The API key is not valid (deleted, rotated or mistyped). Create a new key on the IBM Quantum Platform "
     "dashboard and run save_account.py again."),
    ("could not be found",
     "The API key could not be found (deleted or rotated). Create a new key and run save_account.py again."),
    ("instance",
     "The saved instance is not visible to this API key. Copy the CRN from the Instances page (Open Plan instances "
     "are in us-east) and run save_account.py again with --instance."),
]


def redacted(account: dict) -> dict:
    keep = ("channel", "instance", "region", "plans_preference", "tags")
    shown = {key: account[key] for key in keep if account.get(key)}
    shown["token"] = "<saved>" if account.get("token") else "<missing>"
    return shown


def diagnose(text: str) -> str:
    for needle, advice in DIAGNOSES:
        if needle.lower() in text.lower():
            return advice
    return "Unrecognised error: check network access and the account on https://quantum.cloud.ibm.com."


def main() -> int:
    parser = argparse.ArgumentParser(description="Check IBM Quantum access (no QPU time used).")
    parser.add_argument("--name", help="saved account name (default: the default saved account)")
    parser.add_argument("--filename", help="credentials file (default: ~/.qiskit/qiskit-ibm.json)")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    from qiskit_ibm_runtime import QiskitRuntimeService

    try:
        saved = QiskitRuntimeService.saved_accounts(filename=args.filename)
    except Exception:
        saved = {}
    if not saved:
        print("NO SAVED ACCOUNT. Run save_account.py yourself, in a terminal (it prompts for the API key hidden).")
        return 2
    print("Saved accounts:")
    for name, account in saved.items():
        print(f"  {name}: {redacted(account)}")
    if args.name and args.name not in saved:
        print(f"No saved account named {args.name!r}.")
        return 2

    try:
        kwargs = {"filename": args.filename} if args.filename else {}
        service = QiskitRuntimeService(name=args.name, **kwargs) if args.name else QiskitRuntimeService(**kwargs)
    except Exception as error:
        text = f"{type(error).__name__}: {error} {error.__cause__ or ''}"
        print(f"ACCESS FAILED for {args.name or 'the default account'}: {text.strip()}")
        print(f"What to do: {diagnose(text)}")
        return 1

    active = service.active_account() or {}
    print(f"Connected. Instance: {active.get('instance', '?')}")
    try:
        usage = service.usage()
        if "usage_limit_seconds" in usage:
            used, limit = usage.get("usage_consumed_seconds", 0), usage["usage_limit_seconds"]
            remaining = usage.get("usage_remaining_seconds", limit - used)
            window_end = (usage.get("usage_period") or {}).get("end_time", "?")
            print(f"QPU time: {used} s used of {limit} s ({remaining} s = {remaining / 60:.1f} min left); "
                  f"current window ends {window_end}; limit reached: {usage.get('usage_limit_reached')}")
        else:
            print("Usage:", json.dumps(usage, default=str))
    except Exception as error:
        print(f"Usage not available: {type(error).__name__}: {error}")
    print("Backends:")
    for backend in sorted(service.backends(), key=lambda b: b.name):
        status = backend.status()
        print(f"  {backend.name:18s} {backend.num_qubits:4d} qubits  operational={status.operational}  "
              f"pending_jobs={status.pending_jobs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
