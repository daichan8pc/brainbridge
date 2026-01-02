import streamlit as st
import time
import pandas as pd
import torch
import numpy as np
from brain_net import BrainNet  # あなたのモデル定義ファイル

# --- 1. 画面設定（3.5インチ対応） ---
st.set_page_config(page_title="BrainBridge", layout="wide")

# CSSでデザインを強制改造（ボタンと文字を巨大化）
st.markdown("""
    <style>
    .stApp { background-color: #000000; color: white; }
    
    /* 巨大な感情表示テキスト */
    .result-text {
        font-size: 60px !important;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
        padding: 20px;
        border-radius: 15px;
    }
    
    /* ボタンを画面いっぱいに広げる */
    .stButton > button {
        width: 100%;
        height: 100px;
        font-size: 30px !important;
        font-weight: bold;
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        border: none;
    }
    /* 押した後のボタンの色 */
    .stButton > button:active {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. モデル読み込み ---
@st.cache_resource
def load_model_resources():
    try:
        checkpoint = torch.load("brain_model_dl.pkl", map_location=torch.device('cpu'), weights_only=False)
        
        model = BrainNet(input_size=checkpoint['input_size'])
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        
        # テストデータも返すように変更
        # .get() を使うと、もしキーがなくてもエラーにならず None を返すので安全
        X_test = checkpoint.get('X_test')
        y_test = checkpoint.get('y_test')
        
        return model, checkpoint['scaler'], checkpoint['encoder'], X_test, y_test
    except Exception as e:
        st.error(f"モデル読み込みエラー: {e}")
        return None, None, None, None, None

# 戻り値が増えたので受け取り変数も増やす
model, scaler, encoder, X_test, y_test = load_model_resources()

# --- 3. 脳波データ取得関数（ダミー） ---
# ★重要★ 本番ではここに NeuroSky の取得コードを入れます
def get_realtime_eeg():
    # もしテストデータがあれば、そこからランダムに1つ選ぶ
    if X_test is not None:
        # 0番目〜データの最後までのインデックスをランダムに1つ決める
        random_idx = np.random.randint(0, len(X_test))
        
        # その行のデータを取り出す (1行分の特徴量)
        # ※ X_test はすでに DataFrame か Numpy配列 なので、スライスで取得
        if hasattr(X_test, 'iloc'):
            # DataFrameの場合
            selected_data = X_test.iloc[random_idx].values.reshape(1, -1)
        else:
            # Numpyの場合
            selected_data = X_test[random_idx].reshape(1, -1)
            
        # ★デモ用★ 本当の正解ラベルもこっそり返す（コンソール確認用など）
        true_label = y_test.iloc[random_idx] if hasattr(y_test, 'iloc') else y_test[random_idx]
        
        return selected_data, true_label
        
    else:
        # データがない場合（古いpklなど）は仕方なく乱数
        return np.random.rand(1, 2548), "Unknown"

    
# --- 4. メインアプリのロジック ---

st.markdown("<h3 style='text-align: center;'>BrainBridge AI</h3>", unsafe_allow_html=True)

# セッション状態（画面の記憶）を初期化
if 'result_emotion' not in st.session_state:
    st.session_state['result_emotion'] = None

# --- UI分岐 ---

# A. 結果が出ている場合：結果表示画面
if st.session_state['result_emotion']:
    emotion = st.session_state['result_emotion']
    
    # 色の決定
    color = "#ffffff"
    if "Positive" in emotion: bg_color = "#28a745"  # 緑
    elif "Negative" in emotion: bg_color = "#dc3545" # 赤
    else: bg_color = "#6c757d" # グレー
    
    st.markdown(f"""
        <div style="background-color: {bg_color};" class="result-text">
        {emotion}
        </div>
    """, unsafe_allow_html=True)
    
    st.write("") # 余白
    st.write("") 
    
    # リセットボタン
    if st.button("もう一度計測 (RETRY)"):
        st.session_state['result_emotion'] = None
        st.rerun()

# B. 待機画面：計測スタート
else:
    st.info("装着を確認してボタンを押してください")
    
    if st.button("診断開始 (START)"):
        # --- 計測ループ (5秒間) ---
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        predictions = [] # 結果を溜めるリスト
        
        # 5秒間で約20回計測するループ
        for i in range(20):
            status_text.text(f"脳波解析中... {int((i/20)*100)}%")
            
            # 1. データと、その正解ラベルを取得
            raw_data, true_label = get_realtime_eeg()
            
            # 2. Scalerで正規化
            input_scaled = scaler.transform(raw_data)
            input_tensor = torch.FloatTensor(input_scaled)

            # 3. AI推論
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                
            # 4. 結果をリストに追加（デコードして文字にする）
            pred_label = encoder.inverse_transform(predicted.numpy())[0]
            predictions.append(pred_label)
            
            # プログレスバー更新
            progress_bar.progress((i + 1) / 20)
            time.sleep(0.25) # ウェイト
            
        # --- 最終判定（多数決） ---
        # リストの中で一番多かった感情を選ぶ
        final_decision = max(set(predictions), key=predictions.count)
        
        # 結果を保存して再描画
        st.session_state['result_emotion'] = final_decision
        st.rerun()