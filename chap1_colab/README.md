# 第1章 とにかくDeep Learningを体験！

## Google Colaboratory上で，TensorFlow+Kerasで文字認識（MNISTデータ）をやってみよう．


## 1 Google Colaboratoryを立ち上げる
[Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb)にアクセスする．  
（セキュリティの関係で自動的に別タブ・別ウィンドウには飛ばないようになっている（Tabnabbing対策））  
（右クリックで，<span style="color: red; ">新しいタブ</span>で開いてください）  
googleアカウントを持っていれば，以下のようなページにアクセスできる．  

<img width="1337" alt="colab" src="https://user-images.githubusercontent.com/1255664/72673004-1be28180-3aa7-11ea-958e-4d04ed612c49.png">


## 2 新しいノートブックを作る

左上の「ファイル」から「Python3の新しいノートブック」をクリックして，プログラムを入力するためのノートブックを作成する．

<img width="1320" alt="colab2" src="https://user-images.githubusercontent.com/1255664/72673087-b7282680-3aa8-11ea-8242-7131f18107db.png">

## 3 プログラムを入力する

新しいノートブックができたら，「セル」と呼ばれる四角い窓に，プログラムを入力する．

<img width="935" alt="colab3" src="https://user-images.githubusercontent.com/1255664/72673096-d8891280-3aa8-11ea-951f-ba068ea68eba.png">

次に示すコードが文字（MNIST)を学習して推論するプログラムである．これをコピー＆ペーストする．


```
from __future__ import absolute_import, division, print_function, unicode_literals

try:
  # %tensorflow_version は Colab 上でのみ利用可能
  %tensorflow_version 2.x
except Exception:
  pass

import tensorflow as tf


# MNISTをロードして準備します．サンプルを整数から浮動小数点数に変換します．
# サンプルを整数から浮動小数点数に変換します．
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 層を積み重ねてtf.keras.Sequentialモデルを構築します．
# 訓練のためにオプティマイザと損失関数を選びます．
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(8, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# モデルを学習させる
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)
```


こんな感じになる．
<img width="1179" alt="colab4" src="https://user-images.githubusercontent.com/1255664/72675169-a76a0b80-3ac3-11ea-9240-2255408ecf50.png">


## 4 実行させる

左側にある矢印をクリックするか，「ランタイム」から「すべてのセルを実行」をクリックする．
<img width="1184" alt="colab5" src="https://user-images.githubusercontent.com/1255664/72673732-ebecab80-3ab1-11ea-992b-8a613d92531c.png">

図のように学習し，そして推論結果が出力される．
この場合，正答率が0.9156であった．

## 5 パラメータを変えてみる

次のプログラムの8という数値を128に変更してみると，正答率が変化する．この数値は，ニューラルネットワークの中間層のニューロン数の個数である．これを多くすると学習性能が向上する．その反面，過学習を起こしやすくなり，学習時間も増加する．

```
  tf.keras.layers.Dense(8, activation='relu'),
```
↓

```
  tf.keras.layers.Dense(128, activation='relu'),
```

<img width="1173" alt="colab6" src="https://user-images.githubusercontent.com/1255664/72673840-d1670200-3ab2-11ea-9f3f-42d1c1b0ce99.png">

中間ニューロンの数を8から128に増やすと，正答率が0.9780まであがる．一方，学習時間が各エポックで1秒多くかかっている．

## 6 GPUでもやってみる
GPUも無料に利用できる（利用制限はある）
「ランタイム」から「ランタイムのタイプを変更」を選択

<img width="1168" alt="colab7" src="https://user-images.githubusercontent.com/1255664/72674985-375a8600-3ac1-11ea-8c9b-4e787210a495.png">

「ハードウェア アクセラレータ」を「GPU」を選択
<img width="1178" alt="colab8" src="https://user-images.githubusercontent.com/1255664/72674992-59ec9f00-3ac1-11ea-9158-3a95508f634b.png">

これで，GPUを利用できる．

もう一度，実行してみると，学習速度がNONE（CPU）と比べて早くなっていることがわかる（学習データ，モデルが大規模ではないので，それほどGPUの恩恵を感じないが，）

## 7 自分で書いた文字を認識させてみよう．
どうにか，自分で書いてもらう．
ツールがない人は，[Web](https://www.otwo.jp/blog/canvas-drawing/)を使ってもらう．

## 8 Webアプリで遊んでもらう
[Webデモ](https://ml-demo.zukucode.com/mnist)
[解説](https://zukucode.com/2019/08/tensorflow-vue-mnist.html)
