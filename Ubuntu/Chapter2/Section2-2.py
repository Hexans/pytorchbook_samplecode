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

# Tensorの生成
a = torch.tensor(3, requires_grad=True, dtype=torch.float64)
b = torch.tensor(4, requires_grad=True, dtype=torch.float64)
x = torch.tensor(5, requires_grad=True, dtype=torch.float64)

# 計算グラフの作成 (y = ax + b)
y = a*x + b  # y = 3*5 + 4 = 19
# Tensor yの確認
print(y)

# 変数yに対して微分をし、勾配を算出する
y.backward()

# 勾配を確認
print(a.grad)  # yをaで微分 dy/da = 1*x + 0 = 5
print(b.grad)  # yをbで微分 dy/db = 0 + 1 = 1
print(x.grad)  # yをxで微分 dy/dx = a*1 + 0 = 3
