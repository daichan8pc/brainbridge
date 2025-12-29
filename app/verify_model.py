import pandas as pd
import torch
import pickle
from brain_net import BrainNet

# --- 設定 ---
# 何行目のデータをチェックしたいか（適当に変えてOK）
CHECK_ROW_INDEX = 10 

def verify():
    print(f"--- データの答え合わせ (行番号: {CHECK_ROW_INDEX}) ---")

    # 1. 生データ(CSV)を読み込む（カンニング用）
    df = pd.read_csv('data/emotions.csv')
    
    # 指定した行のデータを取り出す
    row = df.iloc[CHECK_ROW_INDEX]
    
    # 正解ラベル（CSVに書いてある本当の答え）
    true_label = row['label']
    print(f"📝 [正解] CSVのラベル: {true_label}")

    # 2. AIモデルを読み込む
    try:
        with open('brain_model_dl.pkl', 'rb') as f:
            checkpoint = pickle.load(f)
    except FileNotFoundError:
        print("エラー: モデルファイルが見つかりません。")
        return

    # モデルの準備
    model = BrainNet(input_size=checkpoint['input_size'])
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    scaler = checkpoint['scaler']
    encoder = checkpoint['encoder']

    # 3. AIに予想させてみる
    # ラベル以外の数値を入力データとして整形
    input_data = row.drop('label')
    
    # 前処理（正規化など）
    input_scaled = scaler.transform([input_data.values])
    input_tensor = torch.FloatTensor(input_scaled)

    # 推論実行！
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)
    
    # 予測結果を文字に戻す
    predicted_label = encoder.inverse_transform(predicted_idx.numpy())[0]
    
    print(f"🤖 [予測] AIの回答    : {predicted_label}")
    print("-" * 40)

    # 4. 判定
    if true_label == predicted_label:
        print("✅ 正解！データとラベルは正しく紐付いています。")
    else:
        print("❌ 不正解...（学習不足か、本当にズレている可能性があります）")

if __name__ == "__main__":
    verify()