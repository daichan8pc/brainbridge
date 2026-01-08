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
    try:
        # パス注意: データが同じフォルダかdataフォルダにあるか確認
        df = pd.read_csv("data/emotions.csv") 
    except FileNotFoundError:
        # 万が一dataフォルダにない場合、カレントを探す
        df = pd.read_csv("emotions.csv")

    X = df.drop(columns=['label'])
    y = df['label']

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # ★重要: テストデータを確保
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    return X_train, X_test, y_train, y_test, encoder

# --- 2. データ水増し ---
def augment_data(X_train, y_train):
    print(f"元データ数: {len(X_train)}")
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
    print(f"水増し後: {len(X_train_aug)}")
    return X_train_aug, y_train_aug

# --- メイン処理 ---
if __name__ == "__main__":
    X_train_raw, X_test_raw, y_train, y_test, encoder = load_data()
    X_train_aug, y_train_aug = augment_data(X_train_raw, y_train)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_aug)
    
    X_tensor = torch.FloatTensor(X_train_scaled)
    y_tensor = torch.LongTensor(y_train_aug)
    
    model = BrainNet(input_size=X_train_raw.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("学習開始...")
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    # ★最重要★ テストデータごと保存
    save_data = {
        'model_state': model.state_dict(),
        'scaler': scaler,
        'encoder': encoder,
        'input_size': X_train_raw.shape[1],
        'X_test': X_test_raw, 
        'y_test': y_test
    }
    torch.save(save_data, "brain_model_dl.pkl")
    print("完了！ brain_model_dl.pkl を生成しました。")