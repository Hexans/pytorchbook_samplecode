# 必要なパッケージのインストール
!pip3 install torch==1.4.0
!pip3 install torchvision==0.5.0
!pip3 install numpy==1.19.0
!pip3 install matplotlib==3.2.2
!pip3 install scikit-learn==0.23.1
!pip3 install seaborn==0.10.1

# "hello, google colaboratory!"とメッセージを出力
!echo 'hello, google colaboratory!'

# 作業ディレクトリのパスを表示
!pwd

# 作業ディレクトリ内のファイル・フォルダの一覧を取得
!ls

# ファイルのアップロード
from google.colab import files
uploaded = files.upload()

# 作業ディレクトリ内のファイル・フォルダの一覧を確認
!ls

# file.txtの中身を出力
!cat file.txt

# sample_dataフォルダ内のファイル・フォルダの一覧を取得
!ls sample_data/

from google.colab import drive
drive.mount('/content/drive')

# Google Driveがマウントできたか確認
!sudo apt-get install tree
!tree /content
