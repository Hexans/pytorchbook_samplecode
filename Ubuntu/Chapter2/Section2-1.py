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
import numpy as np

# listを渡してTensorを生成
x = torch.tensor([1, 2, 3])
print(x)

# listを入れ子にしてTensorを生成
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(x)

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
x.size()  # Tensorの形状を確認

# データ型を指定せずにTensorを生成
x1 = torch.tensor([[1, 2, 3], [4, 5, 6]])

# dtypeを指定して64ビット浮動小数点数型のTensorを生成
x2 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float64)

# torchのメソッドから64ビット浮動小数点数型のTensorを生成
x3 = torch.DoubleTensor([[1, 2, 3], [4, 5, 6]])

# dtypeの確認
x1.dtype  # データ型指定なし
x2.dtype  # データ型指定
x3.dtype  # データ型指定

# 0から9までの1次元Tensorを生成
x = torch.arange(0, 10)
print(x)

# 0から始まって9まで2ずつ増えていく1次元Tensorを生成
x = torch.linspace(0, 9, 5)
print(x)

# 0から1の間の乱数を生成
x = torch.rand(2, 3)
print(x)

# 2x3の零テンソルを生成
x = torch.zeros(2, 3)
print(x)

# 形状が2x3で要素がすべて1のテンソルを生成
x = torch.ones(2, 3)
print(x)

# GPUがある場合、TensorをGPUへ転送
# x = torch.tensor([1, 2, 3]).to('cuda')
# x.device

# ndarrayの生成
array = np.array([[1,2,3],[4,5,6]])
print(array)

# ndarrayからTensorへ変換
tensor = torch.from_numpy(array)
print(tensor)

# Tensorからndarrayへ変換
tensor2array = tensor.numpy()
print(tensor2array)

# インデックスの指定
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(x[1, 2])

# スライスで要素を取得
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(x[1, :])

# 2x3から3x2のTensorに変換
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
x_reshape = x.view(3, 2)
print(x)  # 変換前の2x3のTensor
print(x_reshape)  # 変換後の3x2のTensor

# Tensorとスカラーの四則演算
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float64)
print(x + 2)  # 足し算
print(x - 2)  # 引き算
print(x * 2)  # 掛け算
print(x / 2)  # 足し算

# Tensor同士の四則演算
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float64)
y = torch.tensor([[4, 5, 6], [7, 8, 9]], dtype=torch.float64)
print(x + y)  # 足し算
print(x - y)  # 引き算
print(x * y)  # 掛け算
print(x / y)  # 割り算

# 基本統計量の算出
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float64)
print(torch.min(x))  # 最小値
print(torch.max(x))  # 最大値
print(torch.mean(x))  # 平均値
print(torch.sum(x))  # 合計値

# Tensorから値を取り出す
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float64)
print(torch.sum(x).item())  # 合計値
