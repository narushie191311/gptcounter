# RunPod用 Dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0,1,2,3

# システムパッケージの更新とインストール
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Python3をデフォルトに設定
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

# 作業ディレクトリの設定
WORKDIR /workspace

# 必要なPythonパッケージのインストール
COPY requirements_colab.txt .
RUN pip install --no-cache-dir -r requirements_colab.txt

# Google Drive連携用のgdownをインストール
RUN pip install --no-cache-dir gdown

# リポジトリのクローン
RUN git clone https://github.com/your-username/gptcounter.git /workspace/gptcounter
WORKDIR /workspace/gptcounter

# モデルファイルのダウンロード（必要に応じて）
RUN mkdir -p models_insightface/models/buffalo_l
# 必要に応じてモデルファイルをダウンロード

# 実行権限の付与
RUN chmod +x scripts/*.sh

# 環境変数の設定
ENV PYTHONPATH=/workspace/gptcounter
ENV DISPLAY=:0

# ヘルスチェック用のスクリプト
COPY healthcheck.py /workspace/
RUN chmod +x /workspace/healthcheck.py

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python /workspace/healthcheck.py

# デフォルトコマンド
CMD ["/bin/bash"]
