#!/bin/bash

# ログファイルに出力（デバッグ用）
exec > >(tee -a ~/brainbridge/app.log) 2>&1
echo "--- Starting BrainBridge System: $(date) ---"

# 1. 環境設定
export OPENBLAS_CORETYPE=ARMV8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export DISPLAY=:0

# 2. ディレクトリ移動（★ここが最重要！モデルファイル読み込みに必須）
cd ~/brainbridge/app

# 3. 仮想環境に入る
source venv/bin/activate

# プロキシ設定を消す（学校/寮のネット環境対策）
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

# 4. クリーンアップ（前回終了時のゴミプロセスを消す）
echo "Cleaning up previous processes..."
pkill -f chromium
pkill -f streamlit

# 5. アプリを起動
echo "Starting Streamlit..."
# --server.headless true: ヘッダーなどを消す
# --server.runOnSave false: 本番運用向け
streamlit run main.py --server.port 8501 --server.headless true --server.runOnSave false &
APP_PID=$!

# 6. 起動待ち(サーバーが立ち上がるまで20秒待つ)
echo "Waiting for Streamlit to launch..."
sleep 20

# 7. ブラウザを起動（キオスクモード）
echo "Starting Chromium..."
# --kiosk: 全画面
# --window-position=0,0 : 左上隅(0,0)から表示開始
# --window-size=480,320 : 画面サイズ(480x320)に強制
# --start-fullscreen    : 念押しのフルスクリーン指定
# --incognito: シークレットモード（キャッシュ対策）
# --noerrdialogs: エラーダイアログを出さない
chromium-browser --kiosk --start-fullscreen --window-position=0,0 --window-size=480,320 --incognito --noerrdialogs --disable-infobars http://localhost:8501

# ブラウザが閉じられたら終了
kill $APP_PID