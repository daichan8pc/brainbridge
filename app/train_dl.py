import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from brain_net import BrainNet

# --- 1. データ準備 ---
def load_data():
    # データ読み込み
    try:
        df = pd.read_csv("data/emotions.csv") # パスは環境に合わせて調整
    except FileNotFoundError:
        print("エラー: data/emotions.csv が見つかりません。")
        exit()

    X = df.drop(columns=['label'])
    y = df['label']

    # ラベルを数字に変換 (NEGATIVE->0, NEUTRAL->1, POSITIVE->2)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # データを分割 (学習8割 : テスト2割)
    # ★重要: ここで分けた X_test (生データ) を後で保存します
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    return X_train, X_test, y_train, y_test, encoder

# --- 2. データ水増し (Augmentation) ---
def augment_data(X_train, y_train):
    print(f"元データ数: {len(X_train)}")
    
    # ノイズを加えてデータを5倍に増やす
    noise_factor = 0.05
    X_augmented = []
    y_augmented = []
    
    for i in range(5):
        noise = np.random.normal(0, noise_factor, X_train.shape)
        X_new = X_train + noise
        X_augmented.append(X_new)
        y_augmented.append(y_train)
    
    X_train_aug = np.vstack(X_augmented)
    y_train_aug = np.concatenate(y_augmented)
    
    print(f"水増し後の学習データ数: {len(X_train_aug)}")
    return X_train_aug, y_train_aug

# --- メイン処理 ---
if __name__ == "__main__":
    # データのロード
    X_train_raw, X_test_raw, y_train, y_test, encoder = load_data()
    
    # 水増し (学習データのみ)
    X_train_aug, y_train_aug = augment_data(X_train_raw, y_train)
    
    # スケーリング (正規化)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_aug)
    
    # Tensor化
    X_tensor = torch.FloatTensor(X_train_scaled)
    y_tensor = torch.LongTensor(y_train_aug)
    
    # モデル初期化
    model = BrainNet(input_size=X_train_raw.shape[1])
    
    # 学習設定
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("ディープラーニング学習開始...")
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # --- 保存処理 (一番大事なところ) ---
    # モデル、Scaler、Encoder、そして「テスト用生データ」を全部まとめて保存
    save_data = {
        'model_state': model.state_dict(),
        'scaler': scaler,
        'encoder': encoder,
        'input_size': X_train_raw.shape[1],
        'X_test': X_test_raw,  # ← アプリでのシミュレーション用に保存
        'y_test': y_test       # ← 答え合わせ用
    }
    
    torch.save(save_data, "brain_model_dl.pkl")
    print("学習完了！ テストデータ付きで 'brain_model_dl.pkl' を保存しました。")