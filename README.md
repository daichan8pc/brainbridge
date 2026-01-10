# BrainBridge - AI Emotion Recognition System

## 1. プロジェクト概要サマリー（二次審査提出用）
BrainBridgeは、感情表現が困難な方のための意思疎通支援デバイスです。Raspberry Piと独自AIモデルを用いて脳波データをリアルタイム解析し、ポジティブ・ネガティブ等の感情を即座に可視化します。ネット不要のスタンドアロン動作と、直感的なUIによるフィードバックを実現。「心」を置き去りにしない、新しいコミュニケーションの形を提案します。（189文字）

## 2. 特徴
* **リアルタイム感情可視化:** 脳波データから感情状態（Positive / Negative / Neutral）を推論し、ディスプレイに表示。
* **エッジAI駆動:** Raspberry Pi 4上で完結する軽量Deep Learningモデル（PyTorch）を実装。外部サーバー不要でプライバシーにも配慮。
* **キオスクモード搭載:** 電源を入れるだけで自動的にアプリが全画面起動するアプライアンス設計。
* **サイバーパンクUI:** 視認性の高いハイコントラストなStreamlitベースのGUI。

## 3. ハードウェア要件
* **Main Board:** Raspberry Pi 4 Model B (4GB or 8GB recommended)
* **Display:** 3.5 inch HDMI LCD Display (480x320 resolution) with Touch support
* **Power:** Mobile Battery (USB-C 5V/3A output)
* **Storage:** MicroSD Card (32GB+)
* **Network:** Wi-Fi (for initial setup and maintenance)

## 4. ソフトウェア要件
* **OS:** Raspberry Pi OS (64-bit / Bullseye or Bookworm)
* **Language:** Python 3.9+
* **Key Libraries:**
    * Streamlit (UI Framework)
    * PyTorch (Deep Learning)
    * Scikit-learn (Preprocessing)
    * Pandas / NumPy (Data manipulation)

## 5. インストール手順
本システムはRaspberry Piのシステムライブラリ（apt）を活用し、仮想環境で動作させます。

### 1. リポジトリのクローン（または配置）
```bash
git clone https://github.com/daichan8pc/BrainBridge.git
cd BrainBridge/app
```

### 2. システム依存関係のインストール

NumPyやPandasの互換性問題を回避するため、apt経由でのインストールを推奨します。

```bash
sudo apt update
sudo apt install -y python3-numpy python3-pandas python3-scikit-learn
```

### 3. 仮想環境の構築

システムパッケージを引き継ぐ設定（`--system-site-packages`）で環境を作成します。

```bash
# 既存の環境がある場合は削除
rm -rf venv

# 仮想環境作成
python3 -m venv venv --system-site-packages

# アクティベート
source venv/bin/activate

# Pythonライブラリのインストール
pip install streamlit torch torchvision
```

### 4. 学習モデルの生成

デモ用の軽量モデルを生成します。

```bash
python3 train_dl.py
# "Model saved..." と表示されれば成功
```

## 6. 実行方法

### 通常起動（開発・テスト用）

```bash
source venv/bin/activate

# 重要: Raspberry Piでのクラッシュ回避用設定
export OPENBLAS_CORETYPE=ARMV8

# アプリケーション起動
streamlit run main.py --server.address 0.0.0.0
```

### 自動起動（キオスクモード）

`~/.config/lxsession/LXDE-pi/autostart` に起動スクリプトを記述することで、電源ON時の自動起動が可能です。詳細はドキュメントまたはセットアップ担当者へ確認してください。

## 7. 終了方法

* **ターミナル実行時:** `Ctrl + C` を押してプロセスを停止します。
* **キオスクモード時:** キーボードを接続し、`Alt + F4` で閉じるか、SSH経由で再起動してください。

## 8. ファイル構成

```text
BrainBridge/
└── app/
    ├── main.py             # アプリケーション本体（UI/推論ロジック）
    ├── brain_net.py        # ニューラルネットワーク定義クラス
    ├── train_dl.py         # モデル学習用スクリプト
    ├── check_crash.py      # 環境依存エラー（Illegal instruction）診断用
    ├── brain_model_dl.pkl  # 学習済みモデル（train_dl.pyで生成）
    ├── requirements.txt    # 依存ライブラリ一覧
    ├── data/
    │   └── emotions.csv    # 学習・推論テスト用データセット
    └── venv/               # Python仮想環境（除外対象）
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
(c) 2026 BrainBridge Project Team