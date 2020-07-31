from flask import Flask, request, render_template
from urllib.error import URLError, HTTPError
import os
import predict_stockprice

app = Flask(__name__)  # Flaskのインスタンスを作成


@app.route('/', methods=['GET', 'POST'])  # ルーティング
def index():
    if request.method == 'GET':
        return render_template('get.html')  # 株価入力ページを表示
    elif request.method == 'POST':
        data = request.form["data"].split('\r\n')  # 入力された株価をリストに変換
        data = [float(val) for val in data if val != '']  # 空白のlistを削除

        y_pred, sub_ma = predict_stockprice.main(data)  # 株価の予測を実行
        from datetime import datetime
        datetime = datetime.now().strftime('%Y%m%d%H%M%S')  # 現在の日時を取得
        # 株価予測ページを表示
        return render_template('post.html', y_pred=y_pred, sub_ma=sub_ma, datetime=datetime)


if __name__ == "__main__":
    app.run()  # Webサーバーの立ち上げ
