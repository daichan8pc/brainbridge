# Copyright (c) 2026 BrainBridge Project Team
# Released under the MIT License
# https://opensource.org/licenses/MIT

import streamlit as st
import torch
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from brain_net import BrainNet

# --- 1. 画面設定 & CSS (1画面収束・コンパクト版) ---
st.set_page_config(page_title="BrainBridge", layout="wide")

st.markdown("""
    <style>
    /* 0. 基本設定: カーソルなし、選択不可 */
    * {
        cursor: none !important;
        -webkit-user-select: none;
        user-select: none;
    }

    /* 1. 全体の背景と余白の削除 */
    .stApp {
        background-color: #000000;
        color: #00FF41;
    }
    
    /* ヘッダー・フッター削除 */
    header[data-testid="stHeader"], footer { display: none; }

    /* 余白を極限まで削る（上下左右ギリギリまで使う） */
    .block-container {
        padding: 0.5rem 0.5rem !important;
        max-width: 100%;
    }

    /* 2. タイトルの小型化 */
    h3 {
        font-family: 'Courier New', monospace;
        font-size: 18px !important;
        margin: 0px 0px 5px 0px !important;
        color: #AAAAAA;
        text-align: center;
        border-bottom: 1px solid #333333;
    }

    /* 3. 結果表示ボックス（左側） */
    .result-box {
        border: 2px solid #FFFFFF;
        padding: 5px;
        text-align: center;
        border-radius: 8px;
        margin-bottom: 10px;
        background-color: #111111;
    }
    .result-label {
        font-size: 14px; color: #AAAAAA; margin: 0;
    }
    .result-text {
        font-size: 32px !important; /* 少し小さくしてはみ出し防止 */
        font-weight: bold;
        color: #FFFFFF;
        margin: 0;
        line-height: 1.2;
    }
    
    /* 4. ボタン（共通） */
    .stButton > button {
        width: 100%;
        height: 50px; /*高さを抑える*/
        font-size: 18px !important;
        font-weight: bold;
        background-color: #333333;
        color: white;
        border: 2px solid #00FF41;
        border-radius: 8px;
        margin-top: 5px;
    }
    .stButton > button:active {
        background-color: #00FF41;
        color: black;
        transform: scale(0.98);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. モデル読み込み関数 ---
@st.cache_resource
def load_model():
    try:
        checkpoint = torch.load("brain_model_dl.pkl", map_location=torch.device('cpu'))
        input_size = checkpoint['input_size']
        model = BrainNet(input_size=input_size)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        return model, checkpoint['scaler'], checkpoint['encoder'], checkpoint.get('X_test'), checkpoint.get('y_test')
    except Exception as e:
        st.error(f"System Error: {e}")
        return None, None, None, None, None

# --- 3. メイン処理 ---
def main():
    # タイトル（常時表示、小さく）
    st.markdown("<h3>BrainBridge AI System</h3>", unsafe_allow_html=True)
    # デモモード（公開データセットを用いた学習であることを示す）
    st.markdown("※ Demo Mode: Emulating EEG Input from test dataset.")
    model, scaler, encoder, X_test, y_test = load_model()
    if model is None:
        return

    # セッション状態の管理
    if 'analyzing' not in st.session_state:
        st.session_state.analyzing = False
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'probs' not in st.session_state:
        st.session_state.probs = None

    # --- A. 待機画面 (STARTボタンのみ) ---
    if not st.session_state.analyzing and st.session_state.result is None:
        st.info("System Ready.")
        st.write("") # スペース調整
        st.write("")
        if st.button("START ANALYSIS"):
            st.session_state.analyzing = True
            st.rerun()

    # --- B. 分析中画面 (プログレスバー) ---
    elif st.session_state.analyzing:
        msg_placeholder = st.empty()
        bar = st.progress(0)
        
        # 演出: 脳波取得中...
        msg_placeholder.text("Acquiring EEG Signal...")
        for i in range(100):
            time.sleep(0.02) # 2秒待つ演出
            bar.progress(i + 1)
        
        # 推論実行
        if X_test is not None:
            # ランダムに1つデータを選ぶ
            idx = np.random.randint(0, len(X_test))
            if hasattr(X_test, 'iloc'):
                raw_data = X_test.iloc[idx].values.reshape(1, -1)
            else:
                raw_data = X_test[idx].reshape(1, -1)
            
            input_scaled = scaler.transform(raw_data)
            input_tensor = torch.FloatTensor(input_scaled)
            
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1).numpy()[0] # 確率分布
                _, predicted = torch.max(output, 1)
                
            pred_label = encoder.inverse_transform(predicted.numpy())[0]
            
            # 結果を保存して画面遷移
            st.session_state.result = pred_label
            st.session_state.probs = probs
            st.session_state.analyzing = False
            st.rerun()

    # --- C. 結果表示画面 (2カラムレイアウト・スクロールなし) ---
    elif st.session_state.result is not None:
        
        # 左右に分割 (比率 1:1)
        col1, col2 = st.columns([1, 1])
        
        # --- 左カラム: 結果とRESTARTボタン ---
        with col1:
            # 結果ボックス
            color_map = {"POSITIVE": "#00FF00", "NEGATIVE": "#FF0000", "NEUTRAL": "#FFFF00"}
            res_color = color_map.get(st.session_state.result, "#FFFFFF")
            
            st.markdown(f"""
                <div class="result-box" style="border-color: {res_color};">
                    <p class="result-label">DETECTED EMOTION</p>
                    <p class="result-text" style="color: {res_color};">{st.session_state.result}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # スペース調整
            st.write("") 
            
            # RESTARTボタン
            if st.button("RESTART SYSTEM"):
                st.session_state.result = None
                st.session_state.probs = None
                st.rerun()

        # --- 右カラム: 確率グラフ ---
        with col2:
            probs = st.session_state.probs
            class_names = encoder.classes_ if hasattr(encoder, 'classes_') else ["NEG", "NEU", "POS"]
            
            # グラフ描画 (極小サイズ)
            fig, ax = plt.subplots(figsize=(3, 2.5)) # サイズ調整
            
            # 背景透明
            fig.patch.set_alpha(0)
            ax.patch.set_alpha(0)
            
            # 棒グラフ
            bars = ax.bar(class_names, probs, color='#00FF41', alpha=0.7)
            
            # 値が高いバーだけ色を変える
            max_idx = np.argmax(probs)
            bars[max_idx].set_color(res_color)
            bars[max_idx].set_alpha(1.0)

            # 文字色設定
            ax.tick_params(axis='x', colors='white', labelsize=10)
            ax.tick_params(axis='y', colors='white', labelsize=8)
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylim(0, 1.0)
            ax.set_title("Probability", color='gray', fontsize=10)

            st.pyplot(fig)
            plt.close(fig)

if __name__ == "__main__":
    main()