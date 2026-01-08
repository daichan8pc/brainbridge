import torch
import numpy as np
from brain_net import BrainNet

def verify():
    print("--- 検証開始 ---")
    try:
        # 1. ファイル読み込み
        checkpoint = torch.load("brain_model_dl.pkl", map_location=torch.device('cpu'), weights_only=False)
        print("✅ ファイル読み込み成功")
        
        # 2. 中身のチェック
        model = BrainNet(input_size=checkpoint['input_size'])
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        
        scaler = checkpoint['scaler']
        encoder = checkpoint['encoder']
        X_test = checkpoint.get('X_test')
        
        print(f"✅ モデル復元成功 (入力サイズ: {checkpoint['input_size']})")
        
        if X_test is None:
            print("❌ テストデータが含まれていません！ train_dl.py を実行してください。")
            return

        print(f"✅ テストデータ確認完了 (データ数: {len(X_test)})")
        
        # 3. 試しに1つ推論してみる
        print("\n--- 推論テスト ---")
        # ランダムに1行取得
        if hasattr(X_test, 'iloc'):
            raw_data = X_test.iloc[0].values.reshape(1, -1)
        else:
            raw_data = X_test[0].reshape(1, -1)
            
        # Scaler -> Model
        input_scaled = scaler.transform(raw_data)
        input_tensor = torch.FloatTensor(input_scaled)
        
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            
        pred_label = encoder.inverse_transform(predicted.numpy())[0]
        print(f"AIの判定結果: {pred_label}")
        print("✅ すべて正常です！")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")

if __name__ == "__main__":
    verify()