---
name: ibm-quantum
description: Set up, verify and use IBM Quantum hardware access from this machine (IBM Quantum Platform on IBM Cloud, Open Plan or paid instances, via qiskit-ibm-runtime). Use whenever the user wants to connect to IBM Quantum, run circuits, QAOA or sampling jobs on IBM QPUs (Heron, Nighthawk, any ibm_* backend), check remaining QPU minutes or which backends are available, set up or repair qiskit-ibm-runtime credentials, or estimate what a hardware run will cost, even if they only say "IBM hardware", "run it on the QPU" or "open plan".
---

# IBM Quantum access

One invocation should leave the user with working, verified access, so they never redo setup. Everything
lives outside any project:

- **Python environment:** a conda env, `ibm-quantum` by default (set IBMQ_ENV to change it), with qiskit,
  qiskit-ibm-runtime and qiskit-aer. `scripts/setup_env.sh` creates it.
- **Credentials:** `~/.qiskit/qiskit-ibm.json`, written by qiskit-ibm-runtime when the user runs
  `scripts/save_account.py`.
- **Named accounts:** accounts are saved by name (for example `open` or `promo`), so code chooses one with
  `QiskitRuntimeService(name="open")`.

## When invoked

1. **Make sure the environment exists.** Run `bash <skill-dir>/scripts/setup_env.sh` and note the `PYTHON=` path it
   prints. It is idempotent, so it takes seconds once the env exists. Pass `--upgrade` only when the user wants
   newer releases.
2. **Check access.** Run `<PYTHON> <skill-dir>/scripts/check_access.py`, adding `--name <account>` to check a
   specific account. No QPU time is spent.
   - **Exit 0:** report the instance, its usage and the operational backends with their queue lengths.
   - **Exit 2 (nothing saved):** ask the user to run this themselves, in their own terminal:
     `<PYTHON> <skill-dir>/scripts/save_account.py --name open --default`.
     - It prompts for the IBM Cloud API key without echoing it, and for the instance CRN.
     - The key comes from the IBM Quantum Platform dashboard; the CRN from its Instances page.
     - If the user has downloaded the key as JSON from the dashboard, the script can read it directly, so the key
       never appears in a command line:
       `save_account.py --name open --instance <CRN> --key-file <path to apikey.json> --default`.
       The user can also authorise you to run this form yourself.
     - Ask them not to paste the key into the chat: the conversation is stored, while the hidden prompt keeps
       the key off the screen and out of shell history.
   - **Exit 1 (a saved account fails):** relay the script's diagnosis.
     - **"Blocked":** the IBM Cloud account behind the key is blocked; only the user can resolve that at
       cloud.ibm.com.
     - **Invalid key:** the key was deleted or rotated.
     - **Wrong instance:** the CRN is wrong or in another region; Open Plan instances are us-east only.

     After the fix, they rerun `save_account.py`.
3. **Then help with the actual task**, following `references/running_jobs.md`. It covers connecting by account
   name, local validation on fake backends, transpiling, estimating cost, job and batch submission, error
   suppression, and logging usage.

## Spending QPU minutes

Minutes are scarce. The Open Plan gives 10 per 28-day window, plus a 2026 promotion of 180 minutes over 12 months.
A careless loop can burn a month's allowance in one go, so treat every hardware submission as needing the user's
explicit go-ahead.

- **Before submitting, state the plan and wait for confirmation.** Give the backend, the number of circuits and
  shots, the two-qubit gate count after transpilation, the execution mode, and the estimated QPU seconds.
  Skip the confirmation only when the user has already approved this exact run.
- **Estimate first.**
  - Use the server-side dry run (`Sampler.run(..., dry_run=True)`) when it is allowed.
  - Otherwise use the back-of-envelope rule in the reference.
  - For a new backend or circuit type, run a tiny job first and read `job.usage()`.
- **Validate correctness locally first** with AerSimulator or a fake backend, so hardware minutes go only to
  circuits already known to be right.
- **Execution modes:** the Open Plan cannot use sessions; use job mode or batch mode.
- **Log every job** (job id, backend, shots, QPU seconds, purpose) to a project file such as
  `logs/ibm_jobs.jsonl`, so results trace back to runs and spend.

## Keeping the key safe

- Load accounts by name in code. The key then never appears in source files, notebooks, environment files,
  repositories or the chat.
- `save_account.py` restricts the credentials file to the user (chmod 600). If a check shows the file is readable
  by others, mention it.
- Delete or overwrite saved accounts only when the user asks. `save_account.py` overwrites only the account with
  the name it is given; `QiskitRuntimeService.delete_account(name=...)` removes one.
