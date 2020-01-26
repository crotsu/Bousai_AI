import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.python.keras.optimizers import SGD
import tensorflow as tf

# トレーニングデータ
# XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# モデル設定
model = Sequential()

# 入力層 - 隠れ層
model.add(Dense(input_dim=2, units=2))
model.add(Activation('sigmoid'))

# 隠れ層 - 出力層
model.add(Dense(units=1))
model.add(Activation('linear'))

sgd = tf.keras.optimizers.SGD(lr=0.1)
model.compile(loss='mean_squared_error', optimizer=sgd)

# モデル学習
model.fit(X, Y, epochs=4000)

# 学習結果の確認
print(model.predict(X))
