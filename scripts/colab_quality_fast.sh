#!/usr/bin/env bash
set -euo pipefail

# Colab(A100)向け 高品質・高速・強レジューム実行ラッパ
# 特徴:
#  - 品質優先: 高解像/毎フレーム検出/高信頼度/StrongSORT
#  - 高速化: FP16, onnxruntime-gpu, 並列シャーディング, マルチプロセス, オートチューニング
#  - 強レジューム: chunk単位CSVを都度フラッシュ/追記、latestリンクとprogressを活用

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

# 引数
VIDEO_PATH=${1:-"/content/merged_20250816_1141-1951.mkv"}
BASE_OUT=${2:-"outputs/analysis_colab.csv"}
EXTRA_ARGS=${3:-""}

mkdir -p outputs logs

LOG_FILE="logs/colab_quality_fast_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[INFO] Colab quality+fast start"
python -V
pip install -U pip || true
pip install -r requirements_colab.txt || true

# gdownでGoogle Driveから取得（既知IDのみ自動、その他は手動配置）
if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "[INFO] video not found, trying to download via gdown..."
  mkdir -p "$(dirname "$VIDEO_PATH")"
  case "$VIDEO_PATH" in
    *20250816*) gdown "https://drive.google.com/uc?id=1spl5lsRrz4hIo-UVr10lgesum6-kUHir" -O "$VIDEO_PATH" || true ;;
    *20250817*) gdown "https://drive.google.com/uc?id=1A_Ai89o9NOT7SgohO4Li2Dh0afuS3SWc" -O "$VIDEO_PATH" || true ;;
    *) echo "[WARN] unknown video path; please ensure the file exists: $VIDEO_PATH" ;;
  esac
fi

python - <<'PY'
import torch
print(f"CUDA: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
  print(torch.cuda.get_device_name(0))
PY

# 高品質既定（必要に応じてEXTRA_ARGSで上書き）
QUAL_ARGS=(
  --device cuda
  --det-size 1280x1280
  --detect-every-n 1
  --conf 0.7
  --body-conf 0.6
  --w-face 0.85
  --w-body 0.15
  --tracker strongsort
  --no-show
  --log-every-sec 10
  --checkpoint-every-sec 15
  --merge-every-sec 60
  --flush-every-n 20
)

# 並列シャーディング（VRAM/ウォームアップ基準で自動シャード）
# 中断耐性: chunk CSVを逐次追記、既存範囲を検出してスキップ

EXTRA=( $EXTRA_ARGS )

python scripts/parallel_shard.py \
  --video "$VIDEO_PATH" \
  --base-output "$BASE_OUT" \
  --target-wall-min 60 \
  --warmup-sec 30 \
  --chunk-sec 900 \
  --tail-chunk-sec 300 \
  --skip-existing 1 \
  --extra-args "${QUAL_ARGS[*]} ${EXTRA[*]}" | cat

echo "[DONE] See outputs and logs. Log: $LOG_FILE"


