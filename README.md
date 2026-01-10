# BrainBridge - EEG Emotion Recognition System

**BrainBridge** は、脳波（EEG）を解析し、リアルタイムで感情（Positive / Negative / Neutral）を推定・可視化するシステムです。
Raspberry Pi 4 上で動作し、ディープラーニングモデル（PyTorch）とタッチパネルUI（Streamlit）を組み合わせたスタンドアローンデバイスとして設計されています。

**Team:** BrainBridge Project Team (Numazu KOSEN & Nara KOSEN)
**Event:** DCON2026 (Deep Learning Contest 2026) Project

## System Architecture

### Hardware
- **Device:** Raspberry Pi 4 Model B (4GB/8GB RAM recommended)
- **Display:** 3.5 inch Touch LCD (or HDMI Display)
- **Sensor:** EEG Headset (connection via LSL/Serial)

### Software Stack
- **OS:** Raspberry Pi OS Legacy (Bullseye) 64-bit
- **Language:** Python 3.9+
- **GUI Framework:** Streamlit (Kiosk Mode)
- **AI Engine:** PyTorch (Neural Network)
- **Visualization:** Matplotlib (Optimized for ARM architecture)

## Installation Guide

本システムは、Raspberry Pi の ARMアーキテクチャに最適化するため、**OS標準ライブラリとPython仮想環境を併用する「ハイブリッド構成」**を採用しています。
再現性を確保するため、以下の手順に従って環境を構築してください。

### 0. OS Preparation (Crucial)
本システムは **Raspberry Pi OS (Bullseye)** での動作を前提としています。
新しいOS（Bookworm等）ではPythonの仮想環境ポリシーが異なるため、以下の特定バージョンのOSイメージを使用することを強く推奨します。

1. **OSイメージのダウンロード**
   以下のリンクまたはアーカイブから、`2024-07-04-raspios-bullseye-arm64.img` をダウンロードしてください。
   * **Version:** Raspberry Pi OS Legacy (64-bit) Bullseye
   * **Release Date:** 2024-07-04

2. **SDカードへの書き込みと初期設定**
   Raspberry Pi Imager 等を使用して、ダウンロードした `.img` ファイルをSDカードに書き込んでください。
   ※書き込み時にOS設定が適用できない場合、以下のいずれかの方法で初期設定（ユーザー作成、Wi-Fi、SSH）を行ってください。

   * **方法A：周辺機器を接続して設定（確実）**
     Raspberry Piにモニター、キーボード、マウスを接続して起動し、画面のウィザードに従ってユーザー名（`pi`推奨）やWi-Fiの設定を完了させてください。

   * **方法B：SDカードに設定ファイルを配置（ヘッドレス）**
     書き込み完了後、PC上でSDカードの `boot` パーティションを開き、直下に以下のファイルを作成することで自動設定が可能です。
     * `ssh` （空ファイル）： SSHを有効化します。
     * `userconf.txt` ： ユーザー自動作成用（形式: `username:encrypted_password`）。
     * `wpa_supplicant.conf` ： Wi-Fi接続情報記述用。

### 1. Prerequisites (System Libraries)
システムレベルで安定した数値計算ライブラリをインストールします。
※ `pip` で入れると `Illegal instruction` エラーが出るため、必ず `apt` を使用してください。

```bash
sudo apt update
sudo apt install -y python3-numpy python3-pandas python3-matplotlib libopenblas-dev libatomic1
```

### 2. Setup Virtual Environment

システムパッケージを引き継ぐ設定（`--system-site-packages`）で仮想環境を作成します。

```bash
cd brainbridge/app
# 既存の環境がある場合は削除: rm -rf venv
python3 -m venv venv --system-site-packages
source venv/bin/activate
```

### 3. Install Dependencies

Python ライブラリをインストールします。
**注意:** `numpy` や `pandas` はインストールせず（システム版を使用）、`streamlit` と `torch` のみを入れます。

```bash
pip install streamlit
# PyTorch (CPU only recommended for Pi)
pip install torch torchvision --extra-index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
```

## Usage

### Manual Start (Debug Mode)

開発やデバッグを行う場合は、以下のスクリプトを使用します。

```bash
cd ~/brainbridge
./run.sh
```

### Kiosk Mode (Auto Start)

電源投入時に自動的に全画面でアプリを起動するには、以下のセットアップを実行してください。

```bash
cd ~/brainbridge/setup
chmod +x install_kiosk.sh
./install_kiosk.sh
sudo reboot
```

設定ファイルは `~/.config/lxsession/LXDE-pi/autostart` に配置されます。

## Project Structure

```text
brainbridge/
├── app/
│   ├── main.py            # アプリケーション本体 (Streamlit)
│   ├── brain_net.py       # AIモデル定義 (PyTorch Class)
│   ├── brain_model_dl.pkl # 学習済みモデル & スケーラー
│   └── train_dl.py        # 学習用スクリプト
├── setup/
│   ├── autostart          # LXDE自動起動設定ファイル
│   └── install_kiosk.sh   # 自動起動インストーラー
├── tools/
│   ├── diagnose.py        # 環境診断ツール (Check libraries)
│   └── verify_model.py    # モデル整合性チェックツール
└── run.sh                 # 統合起動スクリプト
```

## Troubleshooting

**Q. "Illegal instruction" エラーが出る**
A. `pip` でインストールされた `numpy` や `pandas` が ARMv6/v7 用の命令を含んでいる可能性があります。`tools/diagnose.py` を実行して環境を確認し、問題があればシステム版（apt）に入れ替えてください。

**Q. グラフが表示されない / エラーになる**
A. `PyArrow` がインストールされているとクラッシュする場合があります。本システムは `Matplotlib` を使用して描画を行ってください。

## License

[MIT License](https://opensource.org/licenses/MIT)
(c) 2026 BrainBridge Project Team