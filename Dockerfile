# ベースイメージ: Python 3.9 (PyTorch入り)
FROM python:3.9-slim

# 作業ディレクトリ設定
WORKDIR /app

# 必要なシステムパッケージのインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ライブラリのインストール
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ポート開放 (Streamlit用)
EXPOSE 8501

# 実行コマンド
CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0"]