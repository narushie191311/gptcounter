# RunPod用 GPTCounter セットアップ

## 概要
このリポジトリは、RunPodでGPTCounterを実行するための設定ファイルとスクリプトを含んでいます。

## 必要なファイル
- `Dockerfile` - RunPod用のDockerイメージ
- `docker-compose.yml` - マルチGPU対応のDocker Compose設定
- `scripts/setup_runpod.sh` - セットアップスクリプト
- `scripts/download_videos.sh` - Google Drive動画ダウンロード
- `scripts/run_multi_gpu.sh` - マルチGPU実行スクリプト

## セットアップ手順

### 1. リポジトリのクローン
```bash
git clone https://github.com/your-username/gptcounter.git
cd gptcounter
```

### 2. Dockerイメージのビルド
```bash
docker build -t gptcounter-runpod .
```

### 3. コンテナの起動
```bash
docker-compose up -d
```

### 4. コンテナに入る
```bash
docker exec -it gptcounter-runpod bash
```

### 5. セットアップの実行
```bash
bash scripts/setup_runpod.sh
```

### 6. 動画のダウンロード
```bash
bash scripts/download_videos.sh
```

## 実行方法

### 単一GPU実行
```bash
# 16日の動画を実行
python scripts/analyze_video_mac.py \
  --video videos/merged_20250816_1141-1951.mkv \
  --start-sec 0 \
  --output-csv outputs/analysis_16.csv \
  --no-show \
  --device cuda \
  --det-size 1024x1024 \
  --detect-every-n 3 \
  --body-conf 0.5 \
  --conf 0.6 \
  --w-face 0.7 \
  --w-body 0.3 \
  --log-every-sec 15 \
  --checkpoint-every-sec 30 \
  --merge-every-sec 120 \
  --flush-every-n 30 \
  --no-merge \
  --run-id "runpod_16"
```

### マルチGPU実行
```bash
# 複数GPUで並列実行
bash scripts/run_multi_gpu.sh
```

## Google Drive動画ファイル

### 16日の動画
- ファイル名: `merged_20250816_1141-1951.mkv`
- リンク: [Google Drive](https://drive.google.com/file/d/1spl5lsRrz4hIo-UVr10lgesum6-kUHir/view?usp=drive_link)

### 17日の動画
- ファイル名: `merged_20250817_1141-1951.mkv`
- リンク: [Google Drive](https://drive.google.com/file/d/1A_Ai89o9NOT7SgohO4Li2Dh0afuS3SWc/view?usp=drive_link)

## GPU設定

### 環境変数
- `CUDA_VISIBLE_DEVICES=0,1,2,3` - 利用可能なGPUを指定
- `NVIDIA_VISIBLE_DEVICES=all` - すべてのGPUを表示

### GPU確認
```bash
nvidia-smi
python -c "import torch; print(f'GPU数: {torch.cuda.device_count()}')"
```

## トラブルシューティング

### ヘルスチェック
```bash
python healthcheck.py
```

### ログ確認
```bash
docker logs gptcounter-runpod
```

### 権限問題
```bash
chmod +x scripts/*.sh
```

## 注意事項

1. **GPUメモリ**: 大容量動画処理時はGPUメモリを確認
2. **ストレージ**: 出力ファイル用の十分なストレージを確保
3. **ネットワーク**: Google Driveダウンロード時の安定性を確認

## サポート

問題が発生した場合は、以下を確認してください：
- GPUドライバーのバージョン
- CUDAバージョンの互換性
- 利用可能なメモリ容量
- ネットワーク接続状況
