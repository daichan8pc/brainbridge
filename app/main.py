# Copyright (c) 2026 BrainBridge Project Team
# Released under the MIT License
# https://opensource.org/licenses/MIT

import streamlit as st
import torch
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from collections import Counter
from brain_net import BrainNet

# --- 1. 画面設定 & CSS (撮影用・完全版) ---
st.set_page_config(page_title="BrainBridge", layout="wide")

st.markdown("""
    <style>
    /* 0. 基本設定 */
    * { cursor: none !important; -webkit-user-select: none; user-select: none; }
    
    /* 1. 全体の設定 */
    .stApp { background-color: #000000; color: #00FF41; overflow: auto !important; }
    header[data-testid="stHeader"], footer { display: none; }
    
    /* 2. 余白設定 */
    .block-container { 
        padding-top: 0px !important;
        padding-bottom: 5px !important;
        padding-left: 5% !important; 
        padding-right: 5% !important; 
        max-width: 100% !important; 
    }

    /* 3. ボタンの横並び設定 */
    div[data-testid="column"] {
        padding: 0px 5px !important;
    }

    /* 4. タイトル */
    h1 {
        font-family: 'Courier New', monospace; 
        font-size: 36px !important;
        font-weight: bold !important;
        margin: 0px 0px 0px 0px !important;
        color: #FFFFFF;
        text-align: center; 
        border-bottom: 2px solid #00FF41;
        letter-spacing: 2px;
        line-height: 1.0;
        padding-top: 5px;
    }
    
    /* 4.5 デモモード表記 */
    .demo-caption {
        text-align: center;
        font-size: 14px !important;
        color: #FFD700 !important;
        margin-top: 2px;
        margin-bottom: 5px;
        font-family: sans-serif;
        font-weight: bold;
    }
    
    /* 5. 結果ボックス */
    .result-box {
        border: 2px solid #FFFFFF; 
        padding: 2px; 
        text-align: center;
        border-radius: 10px; 
        margin-top: 5px;
        margin-bottom: 10px; 
        background-color: #111111;
        height: 80px; 
        display: flex;
        flex-direction: column;
        justify_content: center;
        align-items: center;
    }
    .result-label { font-size: 10px; color: #AAAAAA; margin: 0; }
    .result-text {
        font-size: 32px !important; 
        font-weight: bold; 
        color: #FFFFFF;
        margin: 0; 
        line-height: 1.0;
    }
    
    /* 6. ボタン設定 */
    .stButton > button {
        width: 100%; 
        height: 45px; 
        font-size: 16px !important;
        font-weight: bold; 
        background-color: #222; 
        color: white;
        border: 2px solid #00FF41; 
        border-radius: 8px; 
        margin: 0px;
        padding: 0px;
    }
    .stButton > button:active { background-color: #00FF41; color: black; transform: scale(0.98); }
    
    /* BACKボタン */
    .back-btn > button {
        border-color: #00AAFF !important;
        color: #00AAFF !important; 
        height: 32px !important;
        font-size: 14px !important;
        margin-bottom: 2px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. モデル読み込み ---
@st.cache_resource
def load_model():
    try:
        checkpoint = torch.load("brain_model_dl.pkl", map_location=torch.device('cpu'))
        model = BrainNet(input_size=checkpoint['input_size'])
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        return model, checkpoint['scaler'], checkpoint['encoder'], checkpoint.get('X_test'), checkpoint.get('y_test')
    except Exception as e:
        st.error(f"System Error: {e}")
        return None, None, None, None, None

# --- 3. センサーデータ取得関数 ---
def fetch_sensor_batch(X_test, samples=10):
    if X_test is not None:
        indices = np.random.randint(0, len(X_test), samples)
        batch_data = []
        for idx in indices:
            if hasattr(X_test, 'iloc'):
                d = X_test.iloc[idx].values.reshape(1, -1)
            else:
                d = X_test[idx].reshape(1, -1)
            batch_data.append(d)
        return batch_data
    return [np.random.rand(1, 2548) for _ in range(samples)]

# --- 4. メイン処理 ---
def main():
    try:
        model, scaler, encoder, X_test, y_test = load_model()
        if model is None: return

        # セッション初期化
        if 'analyzing' not in st.session_state: st.session_state.analyzing = False
        if 'result' not in st.session_state: st.session_state.result = None
        if 'vote_counts' not in st.session_state: st.session_state.vote_counts = None
        if 'show_graph' not in st.session_state: st.session_state.show_graph = False

        # クラッシュ防止
        if st.session_state.show_graph and st.session_state.vote_counts is None:
            st.session_state.show_graph = False
            st.rerun()

        # ★グラフモード画面★
        if st.session_state.show_graph:
            st.markdown('<div class="back-btn">', unsafe_allow_html=True)
            if st.button("BACK"):
                st.session_state.show_graph = False
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

            # グラフ描画
            counts = st.session_state.vote_counts
            class_names = encoder.classes_ if hasattr(encoder, 'classes_') else ["NEG", "NEU", "POS"]
            values = [counts.get(cls, 0) for cls in class_names]
            
            fig, ax = plt.subplots(figsize=(3.5, 1.7)) 
            fig.patch.set_alpha(0); ax.patch.set_alpha(0)
            
            ax.axis('off')
            
            bars = ax.bar(class_names, values, color='#00FF41', alpha=0.7)
            
            color_map = {"POSITIVE": "#00FF00", "NEGATIVE": "#FF0000", "NEUTRAL": "#FFFF00"}
            res_color = color_map.get(st.session_state.result, "#FFFFFF")
            
            max_val = max(values) if values else 1
            ax.set_ylim(0, max_val * 1.3)
            
            plt.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0)

            for bar, val, name in zip(bars, values, class_names):
                if val == max_val and val > 0:
                    bar.set_color(res_color)
                    bar.set_alpha(1.0)
                
                label_text = f"{name}\n({val})"
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.1, label_text, 
                        ha='center', va='bottom', color='white', fontsize=12, fontweight='bold')

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            return

        # --- 以下、通常画面 ---
        st.markdown("<h1>BrainBridge</h1>", unsafe_allow_html=True)
        st.markdown('<div class="demo-caption">※ DEMO MODE: Using Public Dataset</div>', unsafe_allow_html=True)

        # A. 待機中
        if not st.session_state.analyzing and st.session_state.result is None:
            st.write("") 
            st.info("System Ready.")
            st.write("")
            if st.button("START ANALYSIS"):
                st.session_state.analyzing = True
                st.session_state.show_graph = False
                st.rerun()

        # B. 分析中
        elif st.session_state.analyzing:
            msg_placeholder = st.empty()
            bar = st.progress(0)
            msg_placeholder.markdown("<div style='text-align:center; color:#AAA;'>Acquiring Brain Waves...</div>", unsafe_allow_html=True)
            
            batch_data_list = fetch_sensor_batch(X_test, samples=10)
            predictions = []
            
            for i, raw_data in enumerate(batch_data_list):
                progress_val = (i + 1) * 10
                bar.progress(progress_val)
                time.sleep(0.08) 
                
                input_scaled = scaler.transform(raw_data)
                input_tensor = torch.FloatTensor(input_scaled)
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.softmax(output, dim=1).numpy()[0]
                    _, predicted = torch.max(output, 1)
                
                label = encoder.inverse_transform(predicted.numpy())[0]
                predictions.append(label)
                
                # ★ログ出力 (flush=Trueで即時表示)
                print(f"[Input {i+1}/10] Shape:{raw_data.shape} | Probs:{probs} -> Pred:{label}", flush=True)
                
            vote_counts = Counter(predictions)
            final_result, _ = vote_counts.most_common(1)[0]
            
            st.session_state.result = final_result
            st.session_state.vote_counts = vote_counts
            st.session_state.analyzing = False
            st.rerun()

        # C. 結果表示
        elif st.session_state.result is not None:
            color_map = {"POSITIVE": "#00FF00", "NEGATIVE": "#FF0000", "NEUTRAL": "#FFFF00"}
            res_color = color_map.get(st.session_state.result, "#FFFFFF")
            
            st.markdown(f"""
                <div class="result-box" style="border-color: {res_color};">
                    <p class="result-label">DETECTED EMOTION</p>
                    <p class="result-text" style="color: {res_color};">{st.session_state.result}</p>
                </div>
            """, unsafe_allow_html=True)
            
            c_btn1, c_btn2 = st.columns([1, 1])
            
            with c_btn1:
                if st.button("GRAPH"):
                    st.session_state.show_graph = True
                    st.rerun()
            
            with c_btn2:
                if st.button("RESTART"):
                    st.session_state.result = None
                    st.session_state.vote_counts = None
                    st.session_state.show_graph = False
                    st.rerun()
    
    except Exception as e:
        st.error(f"Error: {e}")
        if st.button("Recover"):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main()