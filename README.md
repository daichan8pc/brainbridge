# BrainBridge

**脳波から感情を読み取り、リアルタイムで表現する AI システム**

BrainBridge は、脳波データをディープラーニング（PyTorch）で解析し、現在の感情状態（POSITIVE / NEUTRAL / NEGATIVE）を即座に判定・可視化するプロジェクトです。
現在はプロトタイプ段階として、3.5 インチタッチディスプレイを搭載した Raspberry Pi 上で動作し、事前学習済みデータを用いたリアルタイム・シミュレーションを行います。

## 特徴

- **リアルタイム解析:** 脳波データから瞬時に感情を推論。
- **コンパクトな独立動作:** Raspberry Pi 4B + 3.5 インチ画面だけで完結（スタンドアロン）。
- **高精度 AI モデル:** PyTorch を用いた 3 層ニューラルネットワークを採用。
- **シミュレーションモード:** 実際の脳波計がない環境でも、テストデータを用いてデモ動作が可能。

## ハードウェア要件

- **本体:** Raspberry Pi 4 Model B
- **ディスプレイ:** OSOYOO 3.5 インチ HDMI タッチスクリーン
- **その他:** microSD カード (16GB 以上推奨), 電源

## ソフトウェア要件 (重要)

本システムは、ドライバおよびライブラリの互換性のため、以下の環境を強く推奨します。

- **OS:** Raspberry Pi OS Legacy (Bullseye, 64-bit)
  - ※最新の Bookworm や Trixie では画面ドライバや PyTorch が動作しない可能性があります。
- **Python:** 3.9 (OS 標準)

## インストール手順

### 1. リポジトリのクローン

```bash
cd ~/projects
git clone https://github.com/daichan8pc/BrainBridge.git
cd BrainBridge/app

```

### 2. システムライブラリのインストール

数値計算ライブラリの高速化のため、OpenBLAS などを導入します。

```bash
sudo apt update
sudo apt install -y libopenblas-dev cython3 libatlas-base-dev m4 libblas-dev

```

### 3. 仮想環境の構築

システムパッケージを利用できる設定で Python 仮想環境を作成します。

```bash
# 古い環境があれば削除
rm -rf venv

# --system-site-packages をつけて作成
python3 -m venv venv --system-site-packages
source venv/bin/activate

```

### 4. PyTorch のインストール (Armv7/32-bit 用)

Raspberry Pi 4 (32-bit) に対応した特定のバージョンをインストールします。

```bash
pip install --upgrade pip setuptools wheel
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cpu

```

### 5. その他のライブラリ導入

```bash
pip install -r requirements.txt

```

### 6. ディスプレイ設定 (OSOYOO 3.5 インチ)

```bash
# ドライバのダウンロードと適用
git clone https://github.com/osoyoo/HDMI-show.git
chmod -R 755 HDMI-show
cd HDMI-show/
sudo ./hdmi480320

```

## 実行方法

### 手動起動

```bash
source venv/bin/activate
streamlit run main.py

```

ブラウザで `http://localhost:8501` にアクセスすると表示されます。

### キオスクモード（全画面自動起動）

展示などで使用する場合は、`start_brainbridge.sh` を自動起動設定に追加してください。

## 安全な終了方法

Raspberry Pi は、いきなり電源ケーブルを抜くと **microSD カードが破損し、二度と起動しなくなる（データが飛ぶ）** 恐れがあります。
必ず以下のいずれかの方法でシステムを安全にシャットダウンしてください。

### 1. 外部 PC から終了する場合（推奨）

キオスクモード（全画面）で動作中など、画面操作ができない場合は、PC から SSH 接続して終了コマンドを送ります。

```bash
# SSH接続
ssh pi@raspi3b.local  # (ホスト名は設定に合わせてください)

# シャットダウンコマンド実行
sudo shutdown -h now

```

### 2. キーボード/マウスがある場合

- **手動起動時:** ターミナルで `Ctrl + C` を押してアプリを停止し、`sudo shutdown -h now` を実行します。
- **デスクトップ画面:** 左上のメニュー（ラズベリーアイコン）から「Shutdown」を選択します。

> **Note:** 緑色のアクセスランプ（ACT LED）が完全に点滅しなくなるまで待ってから、電源ケーブルを抜いてください。

## ファイル構成

```text
BrainBridge/
├── app/
│   ├── main.py            # アプリ本体 (Streamlit)
│   ├── brain_net.py       # AIモデル定義 (PyTorch)
│   ├── train_dl.py        # 学習＆データ生成スクリプト
│   ├── verify_model.py    # 動作確認用スクリプト
│   ├── brain_model_dl.pkl # 学習済みモデル (Artifact)
│   └── requirements.txt   # 依存ライブラリ一覧
└── README.md              # ドキュメント

```
