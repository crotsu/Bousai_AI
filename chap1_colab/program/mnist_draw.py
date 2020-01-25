# mnistの画像を表示する

import matplotlib.pyplot as plt

# MNISTデータの表示
W = 1  # 横に並べる個数
H = 1   # 縦に並べる個数
fig = plt.figure(figsize=(H, W))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.05, wspace=0.05)
for i in range(W*H):
    ax = fig.add_subplot(H, W, i + 1, xticks=[], yticks=[])
    ax.imshow(x_test[456].reshape((28, 28)), cmap='gray')

plt.show()
