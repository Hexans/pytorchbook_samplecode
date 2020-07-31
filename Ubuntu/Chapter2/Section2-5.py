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

# 必要なパッケージのインストール
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt

# パッケージのインポート
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt

# ニューラルネットワークの定義
class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


# ハイパーパラメータの定義
N = 64  # バッチサイズ: 64
D_in = 1000  # 入力次元: 1000
H = 100  # 隠れ層次元: 100
D_out = 10  # 出力次元: 10
epoch = 100 # 学習回数

# データの生成
x = torch.rand(N, D_in)  # 入力データ
y = torch.rand(N, D_out)  # 正解値

# ネットワークのロード
net = Net(D_in, H, D_out)

# 損失関数
criterion = nn.MSELoss()

# 最適化関数
optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.99), eps=1e-07)

loss_list = []  # 学習ごとの誤差を格納するリスト
# 学習
for i in range(epoch):
    # データを入力して予測値を計算（順伝播）
    y_pred = net(x)
    # 損失（誤差）を計算
    loss = criterion(y_pred, y)
    print("Epoch: {}, Loss: {:.3f}".format(i+1, loss.item()))  # 誤差を表示
    loss_list.append(loss.item())  # 誤差をリスト化して記録

    # 勾配の初期化
    optimizer.zero_grad()
    # 勾配の計算（逆伝搬）
    loss.backward()
    # パラメータ（重み）の更新
    optimizer.step()

# 結果を図示
plt.figure()
plt.title('Training Curve')  # タイトル
plt.xlabel('Epoch')  # x軸のラベル
plt.ylabel('Loss')  # y軸のラベル
plt.plot(range(1, epoch+1), loss_list)  # 学習回数ごとの誤差をプロット
plt.show()  # プロットの表示
