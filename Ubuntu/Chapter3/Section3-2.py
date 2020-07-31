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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch.nn.functional as F
from torch import optim

# データセットのロード
iris = load_iris()

# データセットの説明
print(iris.DESCR)

# データフレームに変換
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())

# 品種の追加
df['Variety'] = iris.target
df.loc[df['Variety'] == 0, 'Variety'] = 'setosa'
df.loc[df['Variety'] == 1, 'Variety'] = 'versicolor'
df.loc[df['Variety'] == 2, 'Variety'] = 'virginica'
print(df.head())

# 基本統計量の確認
print(df.describe())

# データセットの可視化
sns.pairplot(df, hue='Variety')
plt.show()

# データセットの読み込み
iris = load_iris()
data = iris.data  # 特徴量
label = iris.target  # ラベル（品種）

# 学習データとテストデータを分割
train_data, test_data, train_label, test_label = train_test_split(
    data, label, test_size=0.2)

# 学習データとテストデータのサイズの確認
print("train_data size: {}".format(len(train_data)))
print("test_data size: {}".format(len(test_data)))
print("train_label size: {}".format(len(train_label)))
print("test_label size: {}".format(len(test_label)))

# ndarrayをPyTorchのTensorに変換
train_x = torch.Tensor(train_data)
test_x = torch.Tensor(test_data)
train_y = torch.LongTensor(train_label)  # torch.int64のデータ型に
test_y = torch.LongTensor(test_label)  # torch.int64のデータ型に

# 特徴量とラベルを結合したデータセットを作成
train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)

# ミニバッチサイズを指定したデータローダーを作成
train_batch = DataLoader(
    dataset=train_dataset,  # データセットの指定
    batch_size=5,  # バッチサイズの指定
    shuffle=True,  # シャッフルするかどうかの指定
    num_workers=2)  # コアの数
test_batch = DataLoader(
    dataset=test_dataset,
    batch_size=5,
    shuffle=False,
    num_workers=2)

# ミニバッチデータセットの確認
for data, label in train_batch:
    print("batch data size: {}".format(data.size()))  # バッチの入力データサイズ
    print("batch label size: {}".format(label.size()))  # バッチのラベルサイズ
    break

# ニューラルネットワークの定義
class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# ハイパーパラメータの定義
D_in = 4  # 入力次元: 4
H = 100  # 隠れ層次元: 100
D_out = 3  # 出力次元: 3
epoch = 100  # 学習回数

# ネットワークのロード
# CPUとGPUどちらを使うかを指定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Net(D_in, H, D_out).to(device)
# デバイスの確認
print("Device: {}".format(device))

# 損失関数の定義
criterion = nn.CrossEntropyLoss()

# 最適化関数の定義
optimizer = optim.Adam(net.parameters())

# 損失と正解率を保存するリストを作成
train_loss_list = []  # 学習損失
train_accuracy_list = []  # 学習データの正答率
test_loss_list = []  # 評価損失
test_accuracy_list = []  # テストデータの正答率

# 学習（エポック）の実行
for i in range(epoch):
    # エポックの進行状況を表示
    print('---------------------------------------------')
    print("Epoch: {}/{}".format(i+1, epoch))

    # 損失と正解率の初期化
    train_loss = 0  # 学習損失
    train_accuracy = 0  # 学習データの正答数
    test_loss = 0  # 評価損失
    test_accuracy = 0  # テストデータの正答数

    # ---------学習パート--------- #
    # ニューラルネットワークを学習モードに設定
    net.train()
    # ミニバッチごとにデータをロードし学習
    for data, label in train_batch:
        # GPUにTensorを転送
        data = data.to(device)
        label = label.to(device)

        # 勾配を初期化
        optimizer.zero_grad()
        # データを入力して予測値を計算（順伝播）
        y_pred_prob = net(data)
        # 損失（誤差）を計算
        loss = criterion(y_pred_prob, label)
        # 勾配の計算（逆伝搬）
        loss.backward()
        # パラメータ（重み）の更新
        optimizer.step()

        # ミニバッチごとの損失を蓄積
        train_loss += loss.item()

        # 予測したラベルを予測確率y_pred_probから計算
        y_pred_label = torch.max(y_pred_prob, 1)[1]
        # ミニバッチごとに正解したラベル数をカウント
        train_accuracy += torch.sum(y_pred_label == label).item()

    # ミニバッチの平均の損失と正解率を計算
    batch_train_loss = train_loss / len(train_batch.dataset)
    batch_train_accuracy = train_accuracy / len(train_batch.dataset)
    # ---------学習パートはここまで--------- #

    # ---------評価パート--------- #
    # ニューラルネットワークを評価モードに設定
    net.eval()
    # 評価時の計算で自動微分機能をオフにする
    with torch.no_grad():
        for data, label in test_batch:
            # GPUにTensorを転送
            data = data.to(device)
            label = label.to(device)
            # データを入力して予測値を計算（順伝播）
            y_pred_prob = net(data)
            # 損失（誤差）を計算
            loss = criterion(y_pred_prob, label)
            # ミニバッチごとの損失を蓄積
            test_loss += loss.item()

            # 予測したラベルを予測確率y_pred_probから計算
            y_pred_label = torch.max(y_pred_prob, 1)[1]
            # ミニバッチごとに正解したラベル数をカウント
            test_accuracy += torch.sum(y_pred_label == label).item()
    # ミニバッチの平均の損失と正解率を計算
    batch_test_loss = test_loss / len(test_batch.dataset)
    batch_test_accuracy = test_accuracy / len(test_batch.dataset)
    # ---------評価パートはここまで--------- #

    # エポックごとに損失と正解率を表示
    print("Train_Loss: {:.4f} Train_Accuracy: {:.4f}".format(
        batch_train_loss, batch_train_accuracy))
    print("Test_Loss: {:.4f} Test_Accuracy: {:.4f}".format(
        batch_test_loss, batch_test_accuracy))

    # 損失と正解率をリスト化して保存
    train_loss_list.append(batch_train_loss)
    train_accuracy_list.append(batch_train_accuracy)
    test_loss_list.append(batch_test_loss)
    test_accuracy_list.append(batch_test_accuracy)

# 損失
plt.figure()
plt.title('Train and Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(range(epoch), train_loss_list, color='blue',
         linestyle='-', label='Train_Loss')
plt.plot(range(epoch), test_loss_list, color='red',
         linestyle='--', label='Test_Loss')
plt.legend()  # 凡例

# 正解率
plt.figure()
plt.title('Train and Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(range(1, epoch+1), train_accuracy_list, color='blue',
         linestyle='-', label='Train_Accuracy')
plt.plot(range(1, epoch+1), test_accuracy_list, color='red',
         linestyle='--', label='Test_Accuracy')
plt.legend()

# 表示
plt.show()

# CPUとGPUどちらを使うかを指定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 学習パラメータを保存
torch.save(net.to(device).state_dict(), '3-2_iris_net.pth')

# ハイパーパラメータの定義
D_in = 4  # 入力次元: 4
H = 100  # 隠れ層次元: 100
D_out = 3  # 出力次元: 3

# 保存した学習パラメータを読み込む
net2 = Net(D_in, H, D_out).to(device)
net2.load_state_dict(torch.load('3-2_iris_net.pth', map_location=device))

# ニューラルネットワークを評価モードに設定
net2.eval()
# 推定時の計算で自動微分機能をオフにする
with torch.no_grad():
    # 初期化
    test_accuracy = 0
    for data, label in test_batch:
        # GPUにTensorを転送
        data = data.to(device)
        label = label.to(device)
        # データを入力して予測値を計算（順伝播）
        y_pred_prob = net(data)
        # 予測したラベルを予測確率y_pred_probから計算
        y_pred_label = torch.max(y_pred_prob, 1)[1]
        # ミニバッチごとに正解したラベル数をカウント
        test_accuracy += torch.sum(y_pred_label == label).item()
# ミニバッチの平均の損失と正解率を計算
batch_test_accuracy = test_accuracy / len(test_batch.dataset)
# 正解率を表示
print("Accuracy: {:.3f}".format(test_accuracy / len(test_batch.dataset)))
