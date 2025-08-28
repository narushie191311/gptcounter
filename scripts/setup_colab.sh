#!/usr/bin/env bash
set -euo pipefail

# Colab/A100 向けセットアップ（GPU/ヘッドレス）
python -V
pip install --upgrade pip
pip install -r requirements_colab.txt

echo "Setup done."

