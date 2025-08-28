#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN=${PYTHON_BIN:-python3}
VENVDIR=${VENVDIR:-.venv}

echo "[info] create venv at ${VENVDIR}"
${PYTHON_BIN} -m venv "${VENVDIR}"
source "${VENVDIR}/bin/activate"

echo "[info] upgrade pip/setuptools/wheel"
pip install -U pip setuptools wheel

echo "[info] install requirements"
pip install -r requirements.txt

echo "[ok] venv ready. Activate with: source ${VENVDIR}/bin/activate"

