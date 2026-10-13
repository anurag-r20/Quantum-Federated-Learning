#!/usr/bin/env bash
# Create (once) the Python environment used for IBM Quantum work, and print its python path.
#
# Idempotent: when the environment already has the packages, this only checks and prints, so it is
# cheap to run on every invocation of the skill. Pass --upgrade to move to the latest releases.
#
# Environment variables:
#   IBMQ_ENV   conda environment name   (default: ibm-quantum)
#   IBMQ_PY    Python version on create (default: 3.12)
set -euo pipefail

ENV_NAME="${IBMQ_ENV:-ibm-quantum}"
PY_VERSION="${IBMQ_PY:-3.12}"
PACKAGES=(qiskit qiskit-ibm-runtime qiskit-aer)
UPGRADE=0
[[ "${1:-}" == "--upgrade" ]] && UPGRADE=1

CONDA="$(command -v conda || true)"
[[ -z "$CONDA" && -x "$HOME/miniconda3/bin/conda" ]] && CONDA="$HOME/miniconda3/bin/conda"
if [[ -z "$CONDA" ]]; then
  echo "ERROR: conda not found. Install Miniconda, or set IBMQ_ENV to an existing environment." >&2
  exit 1
fi
PY="$("$CONDA" info --base)/envs/$ENV_NAME/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "Creating conda environment '$ENV_NAME' (Python $PY_VERSION)..." >&2
  "$CONDA" create -y -q -n "$ENV_NAME" "python=$PY_VERSION" >/dev/null
fi

install() {
  if command -v uv >/dev/null 2>&1; then
    uv pip install -q --python "$PY" "$@"
  else
    "$PY" -m pip install -q "$@"
  fi
}

if [[ "$UPGRADE" == 1 ]]; then
  install --upgrade "${PACKAGES[@]}"
elif ! "$PY" -c "import qiskit, qiskit_ibm_runtime, qiskit_aer" >/dev/null 2>&1; then
  echo "Installing ${PACKAGES[*]} into '$ENV_NAME'..." >&2
  install "${PACKAGES[@]}"
fi

"$PY" - <<'EOF'
import sys, qiskit, qiskit_ibm_runtime, qiskit_aer
print(f"env ok: qiskit {qiskit.__version__}, qiskit-ibm-runtime {qiskit_ibm_runtime.__version__}, "
      f"qiskit-aer {qiskit_aer.__version__}")
EOF
echo "PYTHON=$PY"
