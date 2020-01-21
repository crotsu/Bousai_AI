from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense
from keras.utils import np_utils
import numpy as np

# 変数の宣言 --- (1)
classes = 3 # いくつに分類するか
data_size = 100 * 100 * 3 # 縦100×横100×3原色

# データを学習しモデルを評価する
def main():
  # データの読み込み --- (2)
  data = np.load("./Preprocessing/disaster.npz")
  X = data["X"] # --- 画像データ
  y = data["y"] # --- ラベルデータ
  # データを2次元に変形する --- (3)
  X = np.reshape(X, (-1, data_size))
  # 訓練データとテストデータに分割 --- (4)
  X_train, X_test, y_train, y_test = train_test_split(X, y)
  # モデルを訓練し評価 --- (5)
  model = train(X_train, y_train)
  model_eval(model, X_test, y_test)

# モデルを構築しデータを学習する
def train(X, y):
  # モデルの構築 --- (6)
  model = Sequential()
  model.add(Dense(units=64, input_dim=(data_size))) # --- (6.1)
  model.add(Activation('relu')) # --- (6.2)
  model.add(Dense(units=classes)) # --- (6.3)
  model.add(Activation('softmax')) # --- (6.4)
  model.compile(loss='sparse_categorical_crossentropy',
    optimizer='sgd', metrics=['accuracy'])
  model.fit(X, y, epochs=60) # データを学習 --- (7)
  return model

# モデルを評価する 
def model_eval(model, X_test, y_test):
  score = model.evaluate(X_test, y_test) # --- (8)
  print('loss=', score[0])
  print('accuracy=', score[1])
  
if __name__ == "__main__":
  main()

