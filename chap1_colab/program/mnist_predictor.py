# predictor
import tensorflow as tf

# モデルを作る
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 学習済みの重みを読み込む
model.load_weights('mnist_weight')

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
output = model.predict(x_test)

for i in output[:12]:
    print(i.argmax())

# MNISTのテスト画像を表示する
# mnistの画像を表示する
import matplotlib.pyplot as plt

W = 4  # 横に並べる個数
H = 3  # 縦に並べる個数
fig = plt.figure(figsize=(H, W))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.05, wspace=0.05)
for i in range(W*H):
    ax = fig.add_subplot(H, W, i + 1, xticks=[], yticks=[])
    ax.imshow(x_test[i].reshape((28, 28)), cmap='gray')

plt.show()
