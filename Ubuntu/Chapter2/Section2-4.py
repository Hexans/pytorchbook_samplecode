# 必要なパッケージのインストール（必要に応じてインストールしてください）
# pip3 install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
# pip3 install torch==1.4.0
# pip3 install torchvision==0.5.0
# pip3 install numpy==1.19.0
# pip3 install matplotlib==3.2.2
# pip3 install scikit-learn==0.23.1
# pip3 install seaborn==0.10.1
# pip3 install pandas==1.0.5
# pip3 install Flask==1.1.2

# 必要なパッケージをインポート
import torch
from torch import nn

# 2.4.1 バイナリ交差エントロピー損失（nn.BCELoss）
t = torch.rand(1)  # ネットワークが予測したクラスが1である確率
y = torch.tensor(1, dtype=torch.float32).random_(2)  # 正解のクラス（0か1）
criterion = nn.BCELoss()  # 損失関数の設定
loss = criterion(t, y)  # 予測値と正解値との誤差を計算
print(loss)

# 2.4.2 ソフトマックス交差エントロピー損失（nn.CrossEntropyLoss）
t = torch.rand(1, 3)  # ネットワークが予測したクラスの確率
y = torch.tensor([1], dtype=torch.int64).random_(3)  # 正解のクラス（0か1か2）
criterion = nn.CrossEntropyLoss()  # 損失関数の設定
loss = criterion(t, y)  # 予測値と正解値との誤差を計算
print(loss)

# 2.4.3 平均二乗誤差損失（nn.MSELoss）
t = torch.rand(1, 10)  # ネットワークが予測した予測値
y = torch.rand(1, 10)  # 正解値
criterion = nn.MSELoss()  # 損失関数の設定
loss = criterion(t, y)  # 予測値と正解値との誤差を計算
print(loss)

# 2.4.4 平均絶対誤差損失（nn.L1Loss）
t = torch.rand(1, 10)  # ネットワークが予測した予測値
y = torch.rand(1, 10)  # 正解値
criterion = nn.L1Loss()  # 損失関数の設定
loss = criterion(t, y)  # 予測値と正解値との誤差を計算
print(loss)
