# Copyright (c) 2026 BrainBridge Project Team
# Released under the MIT License
# https://opensource.org/licenses/MIT

import os
# 自動判定に任せる
# os.environ['OPENBLAS_CORETYPE'] = 'ARMV8'

import streamlit as st
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 画面設定 ---
st.set_page_config(page_title="BrainBridge", layout="wide")

# CSS設定
st.markdown("""
    <style>
    /* 全体の背景色と文字色 */
    .stApp {
        background-color: #000000;
        color: #00FF41;
    }
    
    /* 1. 上部のヘッダーバーを完全非表示にする */
    header[data-testid="stHeader"] {
        display: none;
    }
    
    /* 2. 上下の余白（パディング）を極限まで削る */
    .block-container {
        padding-top: 0rem !important;   /* 上の隙間をゼロに */
        padding-bottom: 0rem !important; /* 下の隙間もゼロに */
        padding-left: 1rem !important;   /* 左右は少し開けないと見切れる */
        padding-right: 1rem !important;
        max-width: 100%;
    }

    /* 3. フッター（Made with Streamlit）も消す */
    footer {
        display: none;
    }

    /* --- 以下、既存のデザイン設定 --- */
    h1, h2, h3 { color: #FFFFFF; font-family: 'Courier New', monospace; margin-bottom: 0px; }
    
    .result-box {
        border: 2px solid #FFFFFF; padding: 10px; text-align: center;
        border-radius: 10px; margin-top: 5px; margin-bottom: 5px;
    }
    .result-text { font-size: 40px !important; font-weight: bold; color: white; }
    
    .stButton > button {
        width: 100%; height: 60px; font-size: 20px !important;
        background-color: #333333; color: white;
        border: 1px solid #00FF41; border-radius: 5px;
        margin-top: 5px;
    }
    .stButton > button:active { background-color: #00FF41; color: black; }
    </style>
    """,unsafe_allow_html=True)

# --- 2. モデル読み込み関数 ---
@st.cache_resource
def load_model_resources():
    import torch
    from brain_net import BrainNet

    try:
        print("Loading PyTorch model...")
        checkpoint = torch.load("brain_model_dl.pkl", map_location=torch.device('cpu'))
        
        model = BrainNet(input_size=checkpoint['input_size'])
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        
        print("Model loaded successfully!")
        return model, checkpoint['scaler'], checkpoint['encoder'], checkpoint.get('X_test'), checkpoint.get('y_test')
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, None, None

# --- 3. メイン処理 ---
st.markdown("<h2 style='text-align: center;'>BrainBridge AI System</h2>", unsafe_allow_html=True)

# タイトル表示後、スピナーを回しながら読み込みを開始
with st.spinner('システム起動中... AIモデルを読み込んでいます...'):
    model, scaler, encoder, X_test, y_test = load_model_resources()

# 読み込み失敗時のガード
if model is None:
    st.error("モデルの読み込みに失敗しました。")
    st.stop()

# --- 4. 補助関数（torchを使うためここでもimportが必要な場合があるが、ロード済みならOK） ---
def get_realtime_eeg():
    if X_test is not None:
        idx = np.random.randint(0, len(X_test))
        if hasattr(X_test, 'iloc'):
            raw_data = X_test.iloc[idx].values.reshape(1, -1)
        else:
            raw_data = X_test[idx].reshape(1, -1)
        return raw_data
    return np.random.rand(1, 2548)

# --- 5. UIロジック ---
if 'result_emotion' not in st.session_state:
    st.session_state['result_emotion'] = None
    st.session_state['probs'] = None

if st.session_state['result_emotion']:
    emotion = st.session_state['result_emotion']
    probs = st.session_state['probs']
    
    if "POSITIVE" in emotion.upper(): bg = "#28a745"
    elif "NEGATIVE" in emotion.upper(): bg = "#dc3545"
    else: bg = "#6c757d"
    
    st.markdown(f"""
        <div class="result-box" style="background-color: {bg};">
            <div style="font-size: 20px; color: #ddd;">DETECTED EMOTION</div>
            <div class="result-text">{emotion}</div>
        </div>
    """, unsafe_allow_html=True)
    
    if probs is not None:
        st.write("---")
        class_names = encoder.classes_ if hasattr(encoder, 'classes_') else ["NEG", "NEU", "POS"]
        
        # --- Matplotlibで棒グラフを描画（PyArrow回避） ---
        # グラフのサイズ設定
        fig_bar, ax_bar = plt.subplots(figsize=(6, 3))
        
        # 背景を透明にする（アプリの黒背景に馴染ませるため）
        fig_bar.patch.set_alpha(0)
        ax_bar.patch.set_alpha(0)
        
        # 棒グラフを作成（色はBrainBridgeカラーの緑 #00FF41 に合わせる）
        # x軸: class_names, y軸: probs
        # データ長が合わない場合のガード処理も兼ねる
        if len(probs) == len(class_names):
            ax_bar.bar(class_names, probs, color='#00FF41')
        else:
            # 万が一長さが合わない場合はインデックス番号で表示
            ax_bar.bar(range(len(probs)), probs, color='#00FF41')
        
        # 文字色を白にする（背景が黒なので）
        ax_bar.tick_params(axis='x', colors='white')
        ax_bar.tick_params(axis='y', colors='white')
        
        # 枠線の色設定
        ax_bar.spines['bottom'].set_color('white')
        ax_bar.spines['left'].set_color('white')
        
        # 不要な枠線（上と右）を消す
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        
        # タイトル設定
        ax_bar.set_title("Probability Distribution", color='white')

        # Streamlitに表示
        st.pyplot(fig_bar)
        
        # メモリ開放（重要）
        plt.close(fig_bar)

    st.write("")
    if st.button("RESTART SYSTEM"):
        st.session_state['result_emotion'] = None
        st.rerun()

else:
    chart_placeholder = st.empty()
    status_text = st.empty()
    
    if st.button("START ANALYSIS"):
        import torch
        predictions = []
        all_probs = []
        
        for i in range(15):
            status_text.markdown(f"**Acquiring Brain Waves...** ({int((i/15)*100)}%)")
            
            raw_data = get_realtime_eeg()
            input_tensor = torch.FloatTensor(scaler.transform(raw_data))
            
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.nn.functional.softmax(output, dim=1).numpy()[0]
                _, predicted = torch.max(output, 1)
            
            display_wave = raw_data[0][::10] 
            # Matplotlibで描画する方式に変更
            fig, ax = plt.subplots(figsize=(6, 2)) # サイズはお好みで
            ax.plot(display_wave)
            # 余計な枠線を消してスッキリさせる（お好みで）
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Streamlitに渡す
            chart_placeholder.pyplot(fig)

            # メモリリーク防止（ループで回すなら必須）
            plt.close(fig)
            
            predictions.append(predicted.item())
            all_probs.append(prob)
            time.sleep(0.1)
            
        final_idx = max(set(predictions), key=predictions.count)
        final_emotion = encoder.inverse_transform([final_idx])[0]
        final_probs = np.mean(all_probs, axis=0)
        
        st.session_state['result_emotion'] = final_emotion
        st.session_state['probs'] = final_probs
        st.rerun()
    else:
        st.info("System Ready. Press START.")