#!/usr/bin/env bash
# nuReasoning VLA mini-project — environment setup
# Usage: bash scripts/setup_env.sh
set -euo pipefail

ENV_NAME="vla"
PY_VERSION="3.11"

echo "==> Creating conda env '${ENV_NAME}' (python ${PY_VERSION})"
conda create -y -n "${ENV_NAME}" python="${PY_VERSION}"

# Run subsequent pip inside the env without needing `conda activate`
PIP="conda run -n ${ENV_NAME} pip"

echo "==> Installing PyTorch (CUDA 12.4 build)"
${PIP} install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "==> Installing project requirements"
${PIP} install -r requirements.txt

echo "==> Done. Activate with:  conda activate ${ENV_NAME}"
echo "==> Verify with:          bash scripts/check_env.py"
