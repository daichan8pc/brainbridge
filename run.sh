#!/bin/bash

# 1. 環境設定
export OPENBLAS_CORETYPE=ARMV8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 2. 仮想環境に入る
source ~/brainbridge/app/venv/bin/activate
# プロキシ設定を消す
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

# 3. アプリを起動
echo "Starting Streamlit..."
# 既存プロセスをクリーンアップ
pkill -f chromium
pkill -f streamlit
# --server.address 0.0.0.0 に設定
streamlit run ~/brainbridge/app/main.py --server.headless true --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false &
APP_PID=$!

# 4. 起動待ち(20秒待つ)
# 初回はPyTorchの読み込みに時間がかかるため、ここでしっかり待つ
echo "Waiting for 20 seconds..."
sleep 20

# 5. ブラウザを起動
echo "Starting Chromium..."
export DISPLAY=:0
chromium-browser --no-sandbox --disable-gpu --proxy-server="direct://" --proxy-bypass-list="*" --kiosk http://127.0.0.1:8501

# 終了処理
# kill $APP_PID
