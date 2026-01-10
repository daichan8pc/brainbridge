# Copyright (c) 2026 BrainBridge Project Team
# System Diagnostic Tool

import sys
import time

def check_import(module_name):
    print(f"[{module_name}] ... ", end="", flush=True)
    start = time.time()
    try:
        # 動的にインポート
        mod = __import__(module_name)
        # バージョンがあれば表示
        ver = getattr(mod, '__version__', 'N/A')
        elapsed = time.time() - start
        print(f"OK (v{ver}) - {elapsed:.4f}s")
        return True
    except ImportError as e:
        print(f"FAILED! (Not Found)")
        return False
    except Exception as e:
        print(f"CRASHED! ({e})")
        return False

print("=== BrainBridge Environment Diagnosis ===")
print(f"Python Version: {sys.version.split()[0]}")
print("-" * 30)

# 1. 必須ライブラリの生死確認
all_green = True
libraries = [
    "numpy", 
    "pandas", 
    "torch", 
    "streamlit", 
    "matplotlib", # これ重要
    "matplotlib.pyplot"
]

for lib in libraries:
    if not check_import(lib):
        all_green = False

print("-" * 30)

# 2. 自作モジュールの確認
print("[BrainNet Class] ... ", end="", flush=True)
try:
    from brain_net import BrainNet
    model = BrainNet()
    print("OK (Initialized successfully)")
except Exception as e:
    print(f"ERROR: {e}")
    all_green = False

print("-" * 30)

if all_green:
    print("SYSTEM ALL GREEN. Ready to launch!")
else:
    print("SYSTEM ERROR DETECTED.")