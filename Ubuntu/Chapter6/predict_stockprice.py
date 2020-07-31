import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
import torch.nn.functional as F


def preprocessing(close):
    date = range(-49, 1)  # 現在(0日目)から過去50日(-49日目)の日数
    dataset = pd.DataFrame({"date": date, "close": close})
    dataset['25MA'] = dataset['close'].rolling(window=25, min_periods=25).mean()  # 過去50日分のデータから25日移動平均を算出
    test_dataset = dataset[25:]  # 直近の過去25日分のデータを取得

    # 正規化に必要なパラメータ(平均値、標準偏差)を読み込み
    with open(('static/train_files/scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    
    ma = test_dataset['25MA'].values.reshape(-1, 1)  # 二次元配列にreshape
    ma_std = scaler.fit_transform(ma)  # 正規化
    test_x = torch.Tensor(ma_std)  # ndarrayをPyTorchのTensorに変換
    return test_dataset, test_x, scaler


def define_nn():
    # ニューラルネットワークの定義
    class Net(nn.Module):
        def __init__(self, D_in, H, D_out):
            super(Net, self).__init__()
            self.lstm = nn.LSTM(D_in, H, batch_first=True,
                                num_layers=1)
            self.linear = nn.Linear(H, D_out)

        def forward(self, x):
            output, (hidden, cell) = self.lstm(x)
            output = self.linear(output[:, -1, :])  #
            return output

    # # ハイパーパラメータの定義
    D_in = 1  # 入力次元: 1
    H = 200  # 隠れ層次元: 200
    D_out = 1  # 出力次元: 1

    # # CPUとGPUどちらを使うかを指定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # 保存した学習パラメータを読み込む
    net = Net(D_in, H, D_out).to(device)
    net.load_state_dict(torch.load(
        "static/train_files/net.pth", map_location=device))
    return net, device


def predict(net, device, scaler, test_x):
    # ニューラルネットワークを評価モードに設定
    net.eval()
    # 推定時の計算で自動微分機能をオフにする
    with torch.no_grad():
        # GPUにTensorを転送
        data = test_x.to(device)
        # データを入力して予測値を計算（順伝播）
        y_pred = net(data.view(1, -1, 1)).tolist()
    # 正規化の解除
    y_pred = scaler.inverse_transform(y_pred)[0][0]
    y_today = scaler.inverse_transform([test_x.tolist()[-1]])[0][0]
    # 予測した1日後の移動平均と現在の移動平均を比較
    sub_ma = y_pred - y_today
    print('pred_ma: {:.2f}'.format(y_pred))
    print('sub_ma: {:.2f}'.format(sub_ma))
    return round(y_pred, 2), round(sub_ma, 2)  # 小数点第二位で丸める


def plot_result(test_dataset, y_pred):
    # 終値と25日移動平均を図示
    plt.figure()
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.plot(test_dataset['date'], test_dataset['close'], color='black',
             linestyle='-', label="close")
    plt.plot(test_dataset['date'], test_dataset['25MA'], color='dodgerblue',
             linestyle='--', label='true_25MA')
    plt.plot(1, y_pred, color='red', label='predicted_25MA',
             marker='*', markersize=10)
    plt.legend()  # 凡例
    plt.xticks(rotation=30)  # x軸ラベルを30度回転して表示
    plt.savefig("static/predict_files/stock_price_prediction.png")  # 図の保存


def main(data):
    test_dataset, test_x, scaler = preprocessing(data)  # 株価の前処理
    net, device = define_nn()  # ニューラルネットワークの定義とパラメータの読み込み
    y_pred, sub_ma = predict(net, device, scaler, test_x)  # 株価予測
    plot_result(test_dataset, y_pred)  # 結果の図示
    return y_pred, sub_ma


if __name__ == "__main__":
    y_pred, sub_ma = main(data)
