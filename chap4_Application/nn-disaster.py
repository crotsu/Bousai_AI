from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense
from keras.utils import np_utils
import numpy as np

# 乱数の種を初期化
np.random.seed(3) # 自分の環境ではseed=3のとき，accuracy=0.8167

# 変数の宣言
classes = 2 # クラスの数：災害 or 非災害
data_size = 100 * 100 * 3 # 縦100×横100×3原色

# データを学習しモデルを評価する

# データの読み込み
data = np.load("./Data/disaster.npz")
X = data["X"] # 画像データ
y = data["y"] # ラベルデータ

# データを2次元に変形する
X = np.reshape(X, (-1, data_size))

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y)

# モデルを訓練し評価

# モデルの構築
model = Sequential()
model.add(Dense(units=64, input_dim=(data_size)))
model.add(Activation('relu'))
model.add(Dense(units=classes))
model.add(Activation('softmax'))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd', metrics=['accuracy'])

#データを学習
model.fit(X_train, y_train, epochs=60)

# モデルを評価する
score = model.evaluate(X_test, y_test)
print('loss=', score[0])
print('accuracy=', score[1])


# テストデータ
data = np.load("./Data/test.npz")
X = data["X"] # 画像データ
y = data["y"] # ラベルデータ
X = np.reshape(X, (-1, data_size))
print(model.predict(X))
