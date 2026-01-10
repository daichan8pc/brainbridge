# Copyright (c) 2026 BrainBridge Project Team
# Released under the MIT License
# https://opensource.org/licenses/MIT

#!/bin/bash

# スクリプトのディレクトリに移動（どこから実行しても大丈夫なように）
cd "$(dirname "$0")"

# 設定ファイルの配置場所
TARGET_DIR="$HOME/.config/lxsession/LXDE-pi"
TARGET_FILE="$TARGET_DIR/autostart"

echo "--- BrainBridge Kiosk Setup ---"

# 1. ディレクトリがなければ作成
if [ ! -d "$TARGET_DIR" ]; then
    echo "Creating directory: $TARGET_DIR"
    mkdir -p "$TARGET_DIR"
fi

# 2. 既存の設定があればバックアップを取る
if [ -f "$TARGET_FILE" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_FILE="${TARGET_FILE}.backup_${TIMESTAMP}"
    echo "Backing up existing autostart to: $BACKUP_FILE"
    cp "$TARGET_FILE" "$BACKUP_FILE"
fi

# 3. 新しい設定ファイルをコピー
echo "Installing new autostart configuration..."
cp ./autostart "$TARGET_FILE"

# 4. 完了メッセージ
echo "---------------------------------"
echo "Setup Complete!"
echo "The Kiosk mode will be active after the next reboot."
echo "To reboot now, run: sudo reboot"
echo "---------------------------------"