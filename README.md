## ローカル実行(Apple Silicon / Mac Mini M4)

### 前提
- macOS 上の `python3` (推奨: 3.10〜3.12)
- Xcode Command Line Tools が入っていること（通常は不要ですが、無い場合 `xcode-select --install`）

### セットアップ
```bash
chmod +x scripts/setup_venv.sh
scripts/setup_venv.sh
```

手動で行う場合:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

### テスト実行
```bash
.venv/bin/python scripts/local_test_mac.py --device auto --batch-size 128 --train-steps 200
```

`--device` は `auto|mps|cpu` を選択可能です。`auto` は MPS 利用可なら MPS、無ければ CPU を選びます。

## Colab / A100 での利用のヒント
本リポジトリは “Colab 用コードをローカルで軽量動作確認” する目的の最小構成です。A100 向けの最適化（例: CUDA 版 PyTorch、`xformers`、混合精度など）は Colab ノートブック側で設定してください。ローカル(Mac)では CUDA は利用できないため、本テストスクリプトは CPU/MPS で動作します。

### Colab(A100/CUDA) 手順
1) ランタイムを GPU (A100) に変更
2) ノートブック `notebooks/colab_demo.ipynb` を開き、上から実行
   - 依存は `scripts/setup_colab.sh` で自動インストール（`opencv-python-headless` 使用）
   - GUI なしでもエラーにならないよう `--no-show` とヘッドレス検出が既定
3) CLIで実行する場合:
```bash
pip install -r requirements_colab.txt
python scripts/analyze_video_mac.py --video /content/xxx.mkv --start-sec 1800 --duration-sec 60 \
  --output-csv outputs/analysis.csv --video-out outputs/analysis.mp4 --save-video --no-show \
  --device cuda --det-size 1024x1024 --reid-cos 0.6 --gate-iou 0.25 --gate-sim 0.45
python scripts/person_summary.py --input-csv outputs/analysis.csv --video /content/xxx.mkv \
  --output-csv outputs/person_summary.csv --stats-out outputs/person_stats.csv | cat
```


