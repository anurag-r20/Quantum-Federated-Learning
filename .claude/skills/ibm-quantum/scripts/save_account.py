#!/usr/bin/env python3
"""Save IBM Quantum Platform credentials on this machine. Run it yourself, in your own terminal.

The API key is read with getpass, so it is not echoed, does not land in shell history, and never passes
through a chat transcript. qiskit-ibm-runtime writes it to ~/.qiskit/qiskit-ibm.json; this script then
restricts that file to your user (chmod 600) and checks that the key works.

Examples:
    python save_account.py --name open --default          # Open Plan instance, used by default
    python save_account.py --name promo --instance <CRN>  # a second instance, picked by name when needed
    python save_account.py --name open --instance <CRN> --key-file ~/Downloads/apikey.json --default
        # reads the key from the JSON file the IBM dashboard downloads, so it never appears in a command line
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import stat
import sys
import warnings
from pathlib import Path


def read_key_file(path: str) -> str:
    """Returns the API key from IBM's downloaded apikey JSON ({"apikey": ...}) or from a plain text file."""
    text = Path(path).expanduser().read_text().strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text
    label = ", ".join(f"{key}={data[key]}" for key in ("name", "createdAt") if key in data)
    print(f"Read API key from {path} ({label or 'no metadata'}).")
    return str(data.get("apikey", "")).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Save IBM Quantum Platform credentials (API key read hidden).")
    parser.add_argument("--name", default="open", help="name for this saved account, e.g. open or promo (default: open)")
    parser.add_argument("--instance", help="instance CRN or instance name; prompted for when omitted")
    parser.add_argument("--region", choices=["us-east", "eu-de"], help="only used when no instance is given")
    parser.add_argument("--default", action="store_true", help="make this account the default one")
    parser.add_argument("--key-file", help="read the API key from this file (IBM's downloaded apikey JSON, or plain text)")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    from qiskit_ibm_runtime import QiskitRuntimeService

    if args.key_file:
        key = read_key_file(args.key_file)
    else:
        key = getpass.getpass("IBM Cloud API key (input hidden): ").strip()
    if not key:
        print("No API key entered; nothing saved.")
        return 1
    instance = args.instance
    if instance is None:
        instance = input("Instance CRN or name (Instances page on quantum.cloud.ibm.com; blank = let IBM pick): ").strip()

    QiskitRuntimeService.save_account(
        channel="ibm_quantum_platform",
        token=key,
        instance=instance or None,
        region=args.region,
        name=args.name,
        set_as_default=True if args.default else None,
        overwrite=True,  # replaces only the account with this name
    )
    path = Path.home() / ".qiskit" / "qiskit-ibm.json"
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    print(f"Saved account {args.name!r} to {path} (readable by you only).")

    try:
        service = QiskitRuntimeService(name=args.name)
        names = sorted(backend.name for backend in service.backends())
        print(f"Access works: {len(names)} backends: {', '.join(names)}")
    except Exception as error:  # report, do not hide: the key is saved either way
        print(f"Saved, but the access check failed: {type(error).__name__}: {error}")
        print(f"Run check_access.py --name {args.name} for a diagnosis.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
