import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. データの読み込み
# Kaggleのデータセットを読み込みます
df = pd.read_csv('data/emotions.csv')

# 2. データの前処理
# "label" という列が正解（POSITIVEなど）です
# それ以外の列（fft_...など）が学習に使う「特徴量」です
X = df.drop('label', axis=1)  # 正解以外の数値を全部入力にする
y = df['label']               # 正解ラベル

# 3. データを「勉強用」と「テスト用」に分ける
# 全部のデータを勉強に使ってしまうと、実力を測れないので20%は隠しておく
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. モデルの学習 (ここがAIの核心！)
# 「ランダムフォレスト」という、軽量で強力なアルゴリズムを使います
# Jetson Orin Nanoのようなエッジデバイスに最適です
print("学習を開始します... (数秒かかります)")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train) # 勉強用データ(X)と答え(y)を見せて学習させる

# 5. 精度の確認
# 隠しておいたテスト用データで実力テストを行う
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"モデルの精度: {accuracy * 100:.2f}%")
print("これくらいの確率で、脳波から感情を正しく当てられます！")

# 6. モデルの保存
# 学習した「脳みそ」をファイルとして保存します
with open('brain_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("学習完了！ 'brain_model.pkl' が作成されました。")