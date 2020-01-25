# TensorFlowのチュートリアル
# https://www.tensorflow.org/tutorials/quickstart/beginner?hl=ja
# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/ja/tutorials/quickstart/beginner.ipynb

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

# 結果を毎回同じにするために，乱数の種を指定する．
tf.random.set_seed(0)

# MNISTをロードして準備します．サンプルを整数から浮動小数点数に変換します．
# サンプルを整数から浮動小数点数に変換します．
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 層を積み重ねてtf.keras.Sequentialモデルを構築します．
# 訓練のためにオプティマイザと損失関数を選びます．
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# モデルを学習させる
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)

#
# 追加部分
#
# 学習済みのモデルを保存する
# 重みの保存
model.save_weights('mnist_weight')
