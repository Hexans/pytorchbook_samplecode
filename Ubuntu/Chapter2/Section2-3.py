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
import torch.nn.functional as F

# nn.Sequentialで定義
net = torch.nn.Sequential(
    nn.Conv2d(1, 6, 3),  # nn.Conv2d(入力チャネル, 出力チャネル, カーネルサイズ)
    nn.MaxPool2d((2, 2)),  # nn.MaxPool2d(カーネルサイズ)
    nn.ReLU(),
    nn.Conv2d(6, 16, 3),
    nn.MaxPool2d(2),  # nn.MaxPool2d((2,2))と同じ
    nn.ReLU()
)
print(net)

# 自作のクラスを使って定義
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return x

# ネットワークのロード
net = Net()
print(net)

net = Net().to('cuda')
