import streamlit as st
import pandas as pd
import time
import numpy as np
import pickle
import torch
from brain_net import BrainNet # 自作のDLモデル定義をインポート

# ---------------------------------------------------------
# DCON2026 BrainBridge Prototype (Deep Learning Ver.)
# ---------------------------------------------------------

st.set_page_config(page_title="BrainBridge Prototype", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .emotion-box { padding: 20px; border-radius: 10px; text-align: center; color: white; transition: 0.5s; }
    .happy { background-color: #FFB74D; }
    .sad { background-color: #4FC3F7; }
    .neutral { background-color: #90A4AE; }
</style>
""", unsafe_allow_html=True)

# --- データ読み込み ---
@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/emotions.csv")
    except FileNotFoundError:
        return None

# --- AIモデル読み込み (DL版) ---
@st.cache_resource
def load_ai_model():
    try:
        with open('brain_model_dl.pkl', 'rb') as f:
            checkpoint = pickle.load(f)
        
        # モデル構造を復元して重みをロード
        model = BrainNet(input_size=checkpoint['input_size'])
        model.load_state_dict(checkpoint['model_state'])
        model.eval() # 推論モード
        
        return model, checkpoint['scaler'], checkpoint['encoder']
    except FileNotFoundError:
        return None, None, None

def map_emotion(label):
    if label == "POSITIVE": return "Happy", "😊", "happy"
    elif label == "NEGATIVE": return "Sad", "😢", "sad"
    else: return "Relaxed", "🍵", "neutral"

def main():
    st.title("BrainBridge: Emotion Decoder (Deep Learning)")
    st.markdown("### 脳波 × 深層学習 による感情意思伝達")

    # サイドバー
    st.sidebar.header("System Control")
    start_btn = st.sidebar.button("システム起動 (Start)")
    stop_btn = st.sidebar.button("システム停止 (Stop)")
    speed = st.sidebar.slider("更新速度 (秒)", 0.1, 2.0, 1.0)

    df = load_data()
    model, scaler, encoder = load_ai_model()

    if df is None:
        st.error("エラー: data/emotions.csv が見つかりません")
        return
    if model is None:
        st.error("エラー: brain_model_dl.pkl が見つかりません。python3 train_dl.py を実行してください。")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("リアルタイム脳波モニタ")
        chart_placeholder = st.empty()
    with col2:
        st.subheader("AI推定結果")
        emotion_placeholder = st.empty()

    if 'running' not in st.session_state: st.session_state.running = False
    if start_btn: st.session_state.running = True
    if stop_btn: st.session_state.running = False

    if st.session_state.running:
        simulation_stream = df.sample(frac=1).reset_index(drop=True)
        chart_data = []

        for index, row in simulation_stream.iterrows():
            if not st.session_state.running: break

            # 1. 入力データの取得（正解ラベル以外）
            input_raw = row.drop('label')
            
            # グラフ用データ
            chart_data.append(row['fft_0_b'])
            if len(chart_data) > 50: chart_data.pop(0)

            # 2. ディープラーニングによる推論
            # データの正規化
            input_scaled = scaler.transform([input_raw.values])
            input_tensor = torch.FloatTensor(input_scaled)
            
            # ニューラルネットに通す
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted_idx = torch.max(outputs, 1)
            
            # 数字を元のラベル文字に戻す
            prediction_label = encoder.inverse_transform(predicted_idx.numpy())[0]

            # 3. 画面更新
            emotion_text, icon, css_class = map_emotion(prediction_label)
            
            chart_placeholder.line_chart(chart_data)
            emotion_placeholder.markdown(
                f"""
                <div class="emotion-box {css_class}">
                    <h1>{icon}</h1>
                    <h2>{emotion_text}</h2>
                    <p>AI Confidence: High</p>
                </div>
                """, unsafe_allow_html=True
            )
            time.sleep(speed)

if __name__ == "__main__":
    main()