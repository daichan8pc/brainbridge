# Copyright (c) 2026 BrainBridge Project Team
# Released under the MIT License
# https://opensource.org/licenses/MIT

import streamlit as st
import time
import pandas as pd
import torch
import numpy as np
from brain_net import BrainNet

# --- 1. 画面設定 ---
st.set_page_config(page_title="BrainBridge", layout="wide")

# CSSでデザインを調整（黒背景、ハイコントラスト）
st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #00FF41; }
    h1, h2, h3 { color: #FFFFFF; font-family: 'Courier New', monospace; }
    
    /* 結果表示のボックス */
    .result-box {
        border: 2px solid #FFFFFF;
        padding: 20px;
        text-align: center;
        border-radius: 10px;
        margin-top: 10px;
    }
    .result-text {
        font-size: 50px !important;
        font-weight: bold;
        color: white;
    }
    
    /* ボタンのスタイル */
    .stButton > button {
        width: 100%;
        height: 80px;
        font-size: 24px !important;
        background-color: #333333;
        color: white;
        border: 1px solid #00FF41;
        border-radius: 5px;
    }
    .stButton > button:active { background-color: #00FF41; color: black; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. モデル読み込み ---
@st.cache_resource
def load_model_resources():
    try:
        # weights_only=False で読み込み
        checkpoint = torch.load("brain_model_dl.pkl", map_location=torch.device('cpu'), weights_only=False)
        model = BrainNet(input_size=checkpoint['input_size'])
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        return model, checkpoint['scaler'], checkpoint['encoder'], checkpoint.get('X_test'), checkpoint.get('y_test')
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None, None, None

model, scaler, encoder, X_test, y_test = load_model_resources()

# --- 3. データ取得（シミュレーション） ---
def get_realtime_eeg():
    if X_test is not None:
        idx = np.random.randint(0, len(X_test))
        if hasattr(X_test, 'iloc'):
            raw_data = X_test.iloc[idx].values.reshape(1, -1)
        else:
            raw_data = X_test[idx].reshape(1, -1)
        return raw_data
    return np.random.rand(1, 2548)

# --- 4. メインアプリ ---
st.markdown("<h2 style='text-align: center;'>BrainBridge AI System</h2>", unsafe_allow_html=True)

if 'result_emotion' not in st.session_state:
    st.session_state['result_emotion'] = None
    st.session_state['probs'] = None

# === A. 結果表示画面 ===
if st.session_state['result_emotion']:
    emotion = st.session_state['result_emotion']
    probs = st.session_state['probs']
    
    # 色分け
    if "POSITIVE" in emotion.upper(): bg = "#28a745"
    elif "NEGATIVE" in emotion.upper(): bg = "#dc3545"
    else: bg = "#6c757d"
    
    st.markdown(f"""
        <div class="result-box" style="background-color: {bg};">
            <div style="font-size: 20px; color: #ddd;">DETECTED EMOTION</div>
            <div class="result-text">{emotion}</div>
        </div>
    """, unsafe_allow_html=True)
    
    # 確率グラフの表示（もっともらしく見せる）
    if probs is not None:
        st.write("---")
        st.write("Confidence Level:")
        chart_data = pd.DataFrame(probs, index=["NEG", "NEU", "POS"], columns=["Probability"])
        st.bar_chart(chart_data)

    st.write("")
    if st.button("RESTART SYSTEM"):
        st.session_state['result_emotion'] = None
        st.rerun()

# === B. 計測・待機画面 ===
else:
    # 波形表示用の空枠
    chart_placeholder = st.empty()
    status_text = st.empty()
    
    if st.button("START ANALYSIS"):
        predictions = []
        all_probs = []
        
        # 3秒間ループ（演出）
        for i in range(15):
            status_text.markdown(f"**Acquiring Brain Waves...** ({int((i/15)*100)}%)")
            
            # データ取得＆推論
            raw_data = get_realtime_eeg()
            input_tensor = torch.FloatTensor(scaler.transform(raw_data))
            
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.nn.functional.softmax(output, dim=1).numpy()[0]
                _, predicted = torch.max(output, 1)
            
            # グラフ用にデータを間引いて表示（全部だと重いため）
            display_wave = raw_data[0][::10] 
            chart_placeholder.line_chart(display_wave, height=150)
            
            predictions.append(predicted.item())
            all_probs.append(prob)
            time.sleep(0.1)
            
        # 最終判定
        final_idx = max(set(predictions), key=predictions.count)
        final_emotion = encoder.inverse_transform([final_idx])[0]
        final_probs = np.mean(all_probs, axis=0) # 平均確率
        
        st.session_state['result_emotion'] = final_emotion
        st.session_state['probs'] = final_probs
        st.rerun()
    else:
        st.info("System Ready. Press START.")