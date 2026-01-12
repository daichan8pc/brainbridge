#!/bin/bash

echo "=========================================="
echo "   BrainBridge: DEMO RECORDING MODE"
echo "=========================================="

# 1. 環境設定 (run.shと同じ)
export OPENBLAS_CORETYPE=ARMV8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export DISPLAY=:0

# 2. ディレクトリ移動 & venv (run.shと同じ)
cd ~/brainbridge/app
source venv/bin/activate

# プロキシ解除
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

# 3. クリーンアップ
echo "[System] Cleaning up previous processes..."
pkill -f chromium
pkill -f streamlit

# 4. アプリ起動 (Streamlit)
# バックグラウンド(&)で起動しますが、ログは標準出力(この画面)に出ます
echo "[System] Starting AI Engine (Streamlit)..."
streamlit run main.py --server.port 8501 --server.headless true --server.runOnSave false &
STREAMLIT_PID=$!

# 5. 起動待ち
echo "[System] Waiting for engine to initialize (15s)..."
sleep 15

# 6. ブラウザ起動 (Chromium)
# ★ここが重要: '> /dev/null 2>&1' でブラウザの不要なエラーログを全て捨てます
echo "[System] Launching Display (Silent Mode)..."
chromium-browser --kiosk --start-fullscreen --window-position=0,0 --window-size=480,320 --incognito --noerrdialogs --disable-infobars http://localhost:8501 > /dev/null 2>&1 &

echo "=========================================="
echo "   SYSTEM READY. WAITING FOR INPUT..."
echo "=========================================="
echo ""

# 7. プロセス終了待ち
# これがあることで、スクリプトが終了せず、Streamlitからのログを表示し続けます
wait $STREAMLIT_PID