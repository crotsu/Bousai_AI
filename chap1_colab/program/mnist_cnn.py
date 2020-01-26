import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Reshape

# MNISTをロードして準備します．サンプルを整数から浮動小数点数に変換します．
# サンプルを整数から浮動小数点数に変換します．
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# 畳み込みNNモデルを定義します
model = Sequential([
    Reshape((28, 28, 1), input_shape=(28, 28)),
    Conv2D(50, (5, 5), activation='relu'),
    Conv2D(50, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dropout(0.2),
    Dense(100, activation='relu'),
    Dropout(0.4),
    Dense(10, activation='softmax')
])

# モデルをcompileします
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 学習します
model.fit(x_train, y_train, validation_split=0.1, epochs=5, verbose=1)

# テストデータの予測精度を計算します
model.evaluate(x_test, y_test)
