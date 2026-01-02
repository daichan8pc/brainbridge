import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from brain_net import BrainNet # さっき作ったモデルを読み込む
import pickle

# --- 1. データ読み込みと前処理 ---
# 実行時は cd app してから実行することを想定
df = pd.read_csv('data/emotions.csv')

# 数値を扱いやすくする（正規化）
scaler = StandardScaler()
X_raw = df.drop('label', axis=1)
X_scaled = scaler.fit_transform(X_raw)

# 正解ラベル（POSITIVE等）を数字（0, 1, 2）に変換
encoder = LabelEncoder()
y_raw = df['label']
y_encoded = encoder.fit_transform(y_raw)

# --- 2. データ水増し (Data Augmentation) ---
def augment_data(X, y, noise_level=0.1, count=5):
    X_aug, y_aug = [X], [y]
    for _ in range(count):
        noise = np.random.normal(0, noise_level, X.shape)
        X_aug.append(X + noise)
        y_aug.append(y) # ラベルはそのまま
    return np.vstack(X_aug), np.hstack(y_aug)

# 訓練データとテストデータに分割
X_train_val, X_test, y_train_val, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# 訓練データだけ水増しする
X_train_aug, y_train_aug = augment_data(X_train_val, y_train_val)
print(f"元データ数: {len(X_scaled)}")
print(f"水増し後の学習データ数: {len(X_train_aug)}")

# PyTorch用の形式(Tensor)に変換
X_tensor = torch.FloatTensor(X_train_aug)
y_tensor = torch.LongTensor(y_train_aug)

# --- 3. 学習開始 (Training) ---
model = BrainNet(input_size=X_tensor.shape[1])
criterion = nn.CrossEntropyLoss() # 誤差を計算する関数
optimizer = optim.Adam(model.parameters(), lr=0.001) # 最適化手法

print("ディープラーニング学習開始...")
epochs = 100 # 100回繰り返して勉強する
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# --- 4. モデルと設定の保存 ---
save_data = {
    'model_state': model.state_dict(),
    'scaler': scaler,
    'encoder': encoder,
    'input_size': X_tensor.shape[1],
    'X_test': X_test,  # ← 追加：評価用データ(2割)
    'y_test': y_test   # ← 追加：その正解ラベル(答え合わせ用)
}

import torch
torch.save(save_data, "brain_model_dl.pkl")
print("学習完了！ テストデータ付きで 'brain_model_dl.pkl' を保存しました。")