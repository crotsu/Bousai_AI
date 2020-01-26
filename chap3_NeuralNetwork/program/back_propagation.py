# ニューラルネットワークのBP法による学習
# 練習用のため拡張性ゼロのアホアホプログラミング

import numpy as np
import matplotlib.pyplot as plt

# 乱数の種を設定
np.random.seed(3)

# パラメータ
EPSILON = 4.0 # シグモイド関数の傾き
ETA = 0.1     # 学習係数
TIME = 1000   # 学習回数

# シグモイド関数
def sigmoid(x):
    return 1/(1+np.exp(-1*EPSILON*x))

# 入力（XORの入力部分）
dataX = np.array(
    [[0,0],
     [0,1],
     [1,0],
     [1,1]]
)

# 教師信号（XORの出力部分）
dataY = np.array(
    [[0],
     [1],
     [1],
     [0]]
)

# 初期重みと初期閾値をランダムに与える
wab = (np.random.rand()-0.5)*2 * 0.3 # -0.3から0.3の一様乱数
wac = (np.random.rand()-0.5)*2 * 0.3
wbd = (np.random.rand()-0.5)*2 * 0.3
wbe = (np.random.rand()-0.5)*2 * 0.3
wcd = (np.random.rand()-0.5)*2 * 0.3
wce = (np.random.rand()-0.5)*2 * 0.3
tha = (np.random.rand()-0.5)*2 * 0.3
thb = (np.random.rand()-0.5)*2 * 0.3
thc = (np.random.rand()-0.5)*2 * 0.3

# 重みを表示
print("学習前の重み")
print('wab=', wab)
print('wac=', wac)
print('wbd=', wbd)
print('wbe=', wbe)
print('wcd=', wcd)
print('wce=', wce)
print('tha=', tha)
print('thb=', thb)
print('thc=', thc)
print()

# 誤差曲線のグラフ表示用
x = []
y = []

# 学習
for t in range(TIME):

    outall = []
    errorAll = 0.0
    for p in range(len(dataX)):
        #############
        # 前向き計算
        #############

        # 入力層
        outd = dataX[p][0]
        oute = dataX[p][1]

        # 中間層
        xb = wbd * outd + wbe * oute + thb
        outb = sigmoid(xb)

        xc = wcd * outd + wce * oute + thc
        outc = sigmoid(xc)

        # 出力層
        xa = wab * outb + wac * outc + tha
        outa = sigmoid(xa)

        # 誤差計算
        outall.append(outa)
        error = (outa-dataY[p])**2
        errorAll += error

        ##################
        # Back Propagation
        ##################
        deltaa = (outa - dataY[p]) * EPSILON * (1.0 - outa) * outa
        deltab = deltaa * wab * EPSILON * (1.0 - outb) * outb
        deltac = deltaa * wac * EPSILON * (1.0 - outc) * outc

        wab = wab - ETA * deltaa * outb
        wac = wac - ETA * deltaa * outc
        tha = tha - ETA * deltaa

        wbd = wbd - ETA * deltab * outd
        wbe = wbe - ETA * deltab * oute
        thb = thb - ETA * deltab

        wcd = wcd - ETA * deltac * outd
        wce = wce - ETA * deltac * oute
        thc = thc - ETA * deltac

    # 誤差曲線のグラフ表示用の変数
    x.append(t)
    y.append(errorAll)

# 学習後の出力
print("学習後の出力")
print(outall[0])
print(outall[1])
print(outall[2])
print(outall[3])
print()

# 重みを表示
print()
print("学習後の重み")
print('wab=', wab)
print('wac=', wac)
print('wbd=', wbd)
print('wbe=', wbe)
print('wcd=', wcd)
print('wce=', wce)
print('tha=', tha)
print('thb=', thb)
print('thc=', thc)
print()

# 誤差曲線のグラフ表示
plt.plot(x, y)
# 適切な表示範囲を指定
ymin = 0.0
ymax = y[0]
plt.ylim(ymin, ymax)
# グリッド追加
plt.grid(True)
# 表示
plt.show()
