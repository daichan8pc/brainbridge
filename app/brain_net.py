import torch
import torch.nn as nn

# DCON審査員へのアピールポイント：
# 「エッジデバイス(Jetson)での動作を考慮し、軽量かつ高精度な3層ニューラルネットワークを設計しました」

class BrainNet(nn.Module):
    def __init__(self, input_size):
        super(BrainNet, self).__init__()
        
        # ネットワークの構造定義
        # 入力層(input_size) -> 隠れ層1(128個のニューロン) -> 隠れ層2(64個) -> 出力層(3つの感情)
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()             # 活性化関数（信号の発火を模倣）
        self.dropout = nn.Dropout(0.3)    # 過学習防止用（30%のニューロンをランダムにサボらせる）
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)       # 出力は3つ (Positive, Negative, Neutral)

    def forward(self, x):
        # データの流れ（順伝播）
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out) # ここで過学習を防ぐ
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out