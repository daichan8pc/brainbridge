import streamlit as st
import time
import pandas as pd
import torch
import numpy as np
from brain_net import BrainNet

# --- 1. 画面設定（3.5インチ対応） ---
st.set_page_config(page_title="BrainBridge", layout="wide")

# CSSでデザインを強制改造
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
    .stButton > button:active { background-color: #45a049; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. モデル読み込み ---
@st.cache_resource
def load_model_resources():
    try:
        # weights_only=False でセキュリティエラーを回避
        checkpoint = torch.load("brain_model_dl.pkl", map_location=torch.device('cpu'), weights_only=False)
        
        model = BrainNet(input_size=checkpoint['input_size'])
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        
        # テストデータが存在するか確認して取得
        X_test = checkpoint.get('X_test')
        y_test = checkpoint.get('y_test')
        
        return model, checkpoint['scaler'], checkpoint['encoder'], X_test, y_test
    except Exception as e:
        st.error(f"モデル読み込みエラー: {e}")
        return None, None, None, None, None

model, scaler, encoder, X_test, y_test = load_model_resources()

# --- 3. データ取得シミュレーター ---
def get_realtime_eeg():
    """保存されたテストデータからランダムに1行抽出して、リアルな脳波を再現する"""
    if X_test is not None:
        # ランダムなインデックスを決定
        idx = np.random.randint(0, len(X_test))
        
        # データ取り出し (DataFrameかNumpyかで分岐)
        if hasattr(X_test, 'iloc'):
            raw_data = X_test.iloc[idx].values.reshape(1, -1)
            true_label_idx = y_test[idx] if isinstance(y_test, np.ndarray) else y_test.iloc[idx]
        else:
            raw_data = X_test[idx].reshape(1, -1)
            true_label_idx = y_test[idx]
            
        # 参考情報として正解ラベルの文字も返す
        true_label_str = encoder.inverse_transform([true_label_idx])[0]
        return raw_data, true_label_str
    else:
        # データがない場合は乱数 (緊急避難用)
        return np.random.rand(1, 2548), "Unknown"

# --- 4. メインアプリ ---
st.markdown("<h3 style='text-align: center;'>BrainBridge AI</h3>", unsafe_allow_html=True)

# セッション状態初期化
if 'result_emotion' not in st.session_state:
    st.session_state['result_emotion'] = None

# A. 結果表示画面
if st.session_state['result_emotion']:
    emotion = st.session_state['result_emotion']
    
    # 色の決定
    if "POSITIVE" in emotion.upper(): bg_color = "#28a745" # 緑
    elif "NEGATIVE" in emotion.upper(): bg_color = "#dc3545" # 赤
    else: bg_color = "#6c757d" # グレー
    
    st.markdown(f"""
        <div style="background-color: {bg_color};" class="result-text">
        {emotion}
        </div>
    """, unsafe_allow_html=True)
    
    st.write("") 
    if st.button("もう一度計測 (RETRY)"):
        st.session_state['result_emotion'] = None
        st.rerun()

# B. 待機画面
else:
    st.info("装着を確認してボタンを押してください")
    
    if st.button("診断開始 (START)"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        predictions = [] 
        
        # 5秒間の計測シミュレーション (約20回)
        for i in range(20):
            status_text.text(f"解析中... {int((i/20)*100)}%")
            
            # 1. リアルなテストデータを取得
            raw_data, true_debug = get_realtime_eeg()
            
            # 2. 前処理 (Scaler)
            input_scaled = scaler.transform(raw_data)
            input_tensor = torch.FloatTensor(input_scaled)
            
            # 3. AI推論
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                
            pred_label = encoder.inverse_transform(predicted.numpy())[0]
            predictions.append(pred_label)
            
            progress_bar.progress((i + 1) / 20)
            time.sleep(0.1) # 演出用ウェイト
            
        # 最終判定 (多数決)
        final_decision = max(set(predictions), key=predictions.count)
        st.session_state['result_emotion'] = final_decision
        st.rerun()